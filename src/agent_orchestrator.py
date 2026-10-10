"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude Haiku 4.5 - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.

Commands:
  python src/agent_orchestrator.py test            # 3 scenarios, local run, traced to X-Ray
  python src/agent_orchestrator.py chat            # interactive terminal chat
  python src/agent_orchestrator.py deploy          # Tasks 3-6 deployment pipeline (uses the AgentCore CLI)
  python src/agent_orchestrator.py invoke "<msg>"  # call the deployed AgentCore Runtime
  python src/agent_orchestrator.py serve           # HTTP server (what AgentCore Runtime runs)

Deployment uses the AgentCore CLI (`agentcore`, npm package @aws/agentcore,
https://github.com/aws/agentcore-cli) through the pre-written helper
src/agentcore_cli.py - see the README for the prerequisites (Node.js 20+, uv).
"""

import boto3
import json
import time
import os
import sys
import uuid
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)

# ─────────────────────────────────────────────────────
# OBSERVABILITY
# ─────────────────────────────────────────────────────
# `tool` is the Strands @tool decorator wrapped so that every tool call is
# recorded as an X-Ray subsegment (the orchestrator's route_to_* tools become
# the worker-agent nodes on the X-Ray Service Map) and logged at INFO level.
# Use it exactly like `strands.tool`:  @tool  above each tool function.
from agent_observability import (
    tool, tracer, setup_logging, flush_logs, print_trace_hint,
    apply_observability_config, wait_for_runtime_ready,
)


# ─────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────

def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # Create a BedrockModel using the WORKER model
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.1)

    # System prompt for the Inventory Agent
    system_prompt = f"""
    You are an inventory retriever agent. Your ONLY job:
    1. Call check_order_status using customer_id and order_id
    2. Call get_customer_tier and list_customer_orders with the customer_id
    3. Gather order and customer facts from DynamoDB
    4. Report the facts found in JSON format
    Do NOT make decisions. Do NOT add any other commentary.
    """
    

    # Implement check_order_status
    # NOTE: the Orders table has a COMPOSITE key (customer_id = partition key,
    # order_id = sort key), so a get_item needs BOTH values. That is why this
    # tool takes customer_id as well as order_id.
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Look up one order in DynamoDB and report its status, product, dates
        and amount. Reports facts only - it does NOT decide return eligibility.

        Args:
            customer_id: The customer's unique identifier (e.g. CUST-001)
            order_id: The order identifier (e.g. ORD-27176)

        Returns:
            Order record (order_id, status, product_name, order_date, price, ...)
            or a not-found message
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            response = table.get_item(
                Key={
                    'customer_id': customer_id,
                    'order_id': order_id,
                }
            )
            item = response.get('Item')
            if item is None:
                return json.dumps({
                    'found': False,
                    'customer_id': customer_id,
                    'order_id': order_id,
                    'message': 'Order not found for this customer.',
                }, indent=2)
            return json.dumps({'found': True, **item}, indent=2)
        except ClientError as e:
            logger.exception(
                'DynamoDB error looking up order %s for customer %s',
                order_id, customer_id,
            )
            return json.dumps({
                'found': False,
                'customer_id': customer_id,
                'order_id': order_id,
                'error': e.response.get('Error', {}).get('Message', str(e)),
            }, indent=2)

    # Implement get_customer_tier
    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.
        Standard customers have a 30-day return window; Premium customers have 60 days.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            Customer profile including tier and account details
        """
        try:
            table = dynamodb.Table(config.CUSTOMERS_TABLE)
            response = table.get_item(Key={'customer_id': customer_id})
            profile = response.get('Item')
            if profile is None:
                return json.dumps({
                    'found': False,
                    'customer_id': customer_id,
                    'message': 'Customer profile not found.',
                }, indent=2)
            return json.dumps({
                'found': True,
                'customer_id': customer_id,
                'profile': profile,
            }, indent=2, default=str)
        except ClientError as e:
            logger.exception('DynamoDB error looking up customer %s', customer_id)
            return json.dumps({
                'found': False,
                'customer_id': customer_id,
                'error': e.response.get('Error', {}).get('Message', str(e)),
            }, indent=2)


    # Implement list_customer_orders
    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier

        Returns:
            List of all orders with order_id, status, order_date, and amount
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            # Using query to get all items matching the partition key...
            # -> revert back to get_item if fails
            response = table.query(
                KeyConditionExpression=Key('customer_id').eq(customer_id)
            )
            
            # query() returns a list under the plural "Items" key
            orders = response.get("Items", [])
            
            return {
                "found": len(orders) > 0,
                "customer_id": customer_id,
                "orders": orders
            }
            
        except ClientError as e:
            return json.dumps({
                "found": False,
                "customer_id": customer_id,
                "error": e.response.get("Error", {}).get("Message", str(e))
            })

    # Instantiate and return the Agent
    return Agent(model=model, system_prompt=system_prompt, tools=[
        check_order_status,
        get_customer_tier,
        list_customer_orders
        ]
    )


# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────

def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    # Create a BedrockModel
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.1)


    # System prompt for the Refund Agent
    system_prompt = f"""
    You are a customer refund processing agent. Your responsibility is to determine return/refund eligibility based on the order facts provided in WorkflowState.

    Follow these steps in order:
    1. Call get_inventory_context first to retrieve the inventory context from WorkflowState.
    2. Apply the return policy window applicable to the customer's tier.

    Do not make eligibility decisions without consulting the required context and applying the relevant policy.
    Return only the resulting eligibility determination, without additional commentary.
    """
    

    # Implement get_inventory_context
    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier

        Returns:
            The inventory_agent field from WorkflowState, or empty dict if not yet set
        """
        workflow_state = _read_workflow_state(session_id) or {}
        return workflow_state.get("inventory_agent", {})

    # Implement initiate_refund
    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier
            order_id: The order to return
            reason: Customer-provided reason for the return

        Returns:
            Confirmation dict with return_reference number and instructions
        """
        try:
            table = dynamodb.Table(config.ORDERS_TABLE)
            response = table.get_item(
                Key={
                    'customer_id': customer_id,
                    'order_id': order_id,
                }
            )
            item = response.get('Item')
            if item is None:
                return {
                    'success': False,
                    'customer_id': customer_id,
                    'order_id': order_id,
                    'message': 'Order not found for this customer.',
                }

            if not reason or not reason.strip():
                return {
                    'success': False,
                    'customer_id': customer_id,
                    'order_id': order_id,
                    'message': 'A return reason is required to initiate a refund.',
                }

            return_reference = f"RMA-{uuid.uuid4().hex[:8].upper()}"
            timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            table.update_item(
                Key={
                    'customer_id': customer_id,
                    'order_id': order_id,
                },
                UpdateExpression=(
                    'SET return_status = :status, '
                    'return_reason = :reason, '
                    'return_reference = :return_reference, '
                    'return_initiated_at = :timestamp'
                ),
                ExpressionAttributeValues={
                    ':status': 'initiated',
                    ':reason': reason.strip(),
                    ':return_reference': return_reference,
                    ':timestamp': timestamp,
                },
            )

            return {
                'success': True,
                'customer_id': customer_id,
                'order_id': order_id,
                'return_reference': return_reference,
                'status': 'initiated',
                'instructions': (
                    'Your return request has been recorded. Please keep the item and '
                    'packaging ready for the carrier pickup.'
                ),
            }
        except ClientError as e:
            logger.exception(
                'DynamoDB error initiating refund for order %s for customer %s',
                order_id,
                customer_id,
            )
            return {
                'success': False,
                'customer_id': customer_id,
                'order_id': order_id,
                'error': e.response.get('Error', {}).get('Message', str(e)),
            }

    # Instantiate and return the Agent
    return Agent(model=model, system_prompt=system_prompt, tools=[
        get_inventory_context,
        initiate_refund,
    ])


# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────
retrieval_results = {"returns": [], "shipping": [], "warranty": []}

def build_policy_agent() -> Agent:
    """
    Build the Policy Agent - a multi-agent RAG system.

    Internally creates three specialized retriever sub-agents that run in
    PARALLEL, each querying its own Knowledge Base. The coordinator synthesizes
    the combined results into a complete, grounded policy answer.
    """

    # Build ReturnsPolicyRetrieverAgent
    @tool
    def retrieve_returns_policy(query: str) -> str:
        """Retrieve relevant passages from the Returns Policy knowledge base."""
        try:
            passages = retrieve_from_knowledge_base(config.RETURNS_KB_ID, query, top_k=3)
            retrieval_results["returns"] = passages
        except ConnectionError as e:
            retrieval_results["returns"] = []
            return json.dumps({
                "kb": "Returns Policy",
                "error": str(e),
                "passages_found": 0,
            }, indent=2)

        return json.dumps({
            "kb": "Returns Policy",
            "passages_found": len(passages),
            "results": format_kb_results(passages),
        }, indent=2)

    system_prompt = f"""
    You are the Returns Policy retrieval specialist.
    Use the Returns Policy knowledge base tool to answer only with the retrieved passages.
    """
    

    returns_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt=system_prompt,
        tools=[retrieve_returns_policy],
    )

    # Build ShippingPolicyRetrieverAgent
    @tool
    def retrieve_shipping_policy(query: str) -> str:
        """Retrieve relevant passages from the Shipping Policy knowledge base."""
        try:
            passages = retrieve_from_knowledge_base(config.SHIPPING_KB_ID, query, top_k=3)
            retrieval_results["shipping"] = passages
        except ConnectionError as e:
            retrieval_results["shipping"] = []
            return json.dumps({
                "kb": "Shipping Policy",
                "error": str(e),
                "passages_found": 0,
            }, indent=2)

        return json.dumps({
            "kb": "Shipping Policy",
            "passages_found": len(passages),
            "results": format_kb_results(passages),
        }, indent=2)
        
    system_prompt = f"""
    You are the Shipping Policy retrieval specialist.
    Use the Shipping Policy knowledge base tool to answer only with the retrieved passages."""

    shipping_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt=system_prompt,
        tools=[retrieve_shipping_policy],
    )

    # Build WarrantyPolicyRetrieverAgent
    @tool
    def retrieve_warranty_policy(query: str) -> str:
        """Retrieve relevant passages from the Warranty Policy knowledge base."""
        try:
            passages = retrieve_from_knowledge_base(config.WARRANTY_KB_ID, query, top_k=3)
            retrieval_results["warranty"] = passages
        except ConnectionError as e:
            retrieval_results["warranty"] = []
            return json.dumps({
                "kb": "Warranty Policy",
                "error": str(e),
                "passages_found": 0,
            }, indent=2)

        return json.dumps({
            "kb": "Warranty Policy",
            "passages_found": len(passages),
            "results": format_kb_results(passages),
        }, indent=2)
    
    system_prompt=f"""
    You are the Warranty Policy retrieval specialist.
    Use the Warranty Policy knowledge base tool to answer ONLY with the retrieved passages."""

    warranty_retriever = Agent(
        model=BedrockModel(
            model_id=config.WORKER_MODEL_ID,
            region_name=config.AWS_REGION,
            temperature=0.0,
        ),
        system_prompt=system_prompt,
        tools=[retrieve_warranty_policy],
    )

    # Implement search_all_policies - parallel RAG retrieval tool
    @tool
    def search_all_policies(query: str) -> str:
        """
        Query all three policy knowledge bases IN PARALLEL and return combined results.

        Runs ReturnsPolicyRetrieverAgent, ShippingPolicyRetrieverAgent, and
        WarrantyPolicyRetrieverAgent simultaneously, then combines their findings.

        Args:
            query: The customer's policy question

        Returns:
            Combined policy passages from all three knowledge bases
        """
        retrievers = {
            "Returns": returns_retriever,
            "Shipping": shipping_retriever,
            "Warranty": warranty_retriever,
        }

        trace.kb_start({
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        })

        # Define a helper to run one retriever sub-agent
        def _run_retriever(domain: str, agent: Agent, query: str) -> tuple[str, str]:
            """Run one retriever sub-agent and return (domain, result_text)."""
            result = str(agent(query))
            trace.kb_result(domain, result)
            return domain, result

        results: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_map = {
                executor.submit(_run_retriever, domain, agent, query): domain
                for domain, agent in retrievers.items()
            }

            for future in as_completed(future_map):
                domain = future_map[future]
                try:
                    domain_name, result = future.result()
                except Exception as exc:
                    results[domain] = json.dumps({
                        "kb": domain,
                        "error": str(exc),
                        "passages_found": 0,
                    }, indent=2)
                else:
                    results[domain_name] = result

        trace.kb_done(len(retrievers))

        combined = []
        for domain in ["Returns", "Shipping", "Warranty"]:
            part = results.get(domain)
            if part:
                combined.append(f"## {domain} Policy\n{part}")

        return "\n\n".join(combined) if combined else "No relevant policy passages found."


    policy_model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
        temperature=0.2,
    )

    system_prompt = """
    You are the PolicyAgent coordinator for NovaMart customer support.

    Use the search_all_policies tool to retrieve policy passages for any customer question about returns,
    shipping, or warranties. Ground your answer only in the retrieved passages and answer concisely with the
    relevant policy details. If no passage matches, say so clearly.
    """

    return Agent(model=policy_model, system_prompt=system_prompt, tools=[search_all_policies])


# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────

def build_communication_agent() -> Agent:
    """
    Build the Communication Agent.

    Drafts the final customer-facing message by reading the full WorkflowState
    and composing a coherent, empathetic response.
    """

    # Create a BedrockModel
    model = BedrockModel(model_id=config.WORKER_MODEL_ID, region_name=config.AWS_REGION, temperature=0.3)

    # System prompt for the Communication Agent
    system_prompt = f"""
    You are a customer communication agent for Novamart. Your ONLY job:
    1. Read the full WorkFlowState
    2. Include all relevant information from previous agents
    3. Compose ONLY a coherent, empathetic customer-facing message.
    Maintain calm, warm, and professional tone.
    """
    

    # Implement get_full_workflow_context
    @tool
    def get_full_workflow_context(session_id: str) -> dict:
        """
        Read the complete WorkflowState to access all findings from previous agents.

        Args:
            session_id: The current session identifier

        Returns:
            Full WorkflowState dict (inventory_agent, policy_agent, refund_agent)
        """
        workflow_ctx = _read_workflow_state(session_id)
        inventory_agent = workflow_ctx.get("inventory_agent", {})
        policy_agent = workflow_ctx.get("policy_agent", {})
        refund_agent = workflow_ctx.get("refund_agent", {})
        
        return json.dumps({
            "inventory_agent": inventory_agent,
            "policy_agent": policy_agent,
            "refund_agent": refund_agent,
        }, indent=2)
        

    # Instantiate and return the Agent
    return Agent(model=model, system_prompt=system_prompt, tools=[get_full_workflow_context])


# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────

def build_orchestrator_agent(
    inventory_agent:      Agent,
    refund_agent:         Agent,
    policy_agent:         Agent,
    communication_agent:  Agent,
) -> Agent:
    """
    Build the Orchestrator Agent that routes requests and manages WorkflowState.
    """

    # Create a BedrockModel using the ORCHESTRATOR model
    model = BedrockModel(model_id=config.ORCHESTRATOR_MODEL_ID, region_name=config.AWS_REGION, temperature=0.0)

    # System prompt for the Orchestrator
    # For arithmetic, skip Inventory, Policy and Refund, but still call
    # CommunicationAgent last. Round currency only after the full calculation.
    system_prompt = """
    You are an orchestrator agent responsible for managing WorkflowState and routing requests to specialist agents.

    Rules:
    1. Always call initialize_session first for every request.
    2. Route orders, returns, and refunds to the inventory agent first, then the refund agent.
    3. Route policy questions (return windows, shipping rates, warranty terms) to the policy agent.
    4. Route customer-tier questions to the inventory agent.
    5. For each tool call:
        a. Read the current WorkflowState.
        b. Invoke the appropriate specialist.
        c. Update WorkflowState with the result.
    6. Always delegate to the communication agent as the final tool call.
    7. For arithmetic requests, skip the inventory, policy, and refund agents. Complete the full calculation before rounding currency.
    8. Never answer the customer directly or write the final customer-facing response. Always delegate to the appropriate specialist, including the communication agent.
    """
    

    # Each routing tool follows the same pattern:
    #   1. read the current WorkflowState  (_read_workflow_state)
    #   2. invoke the worker agent
    #   3. write its result back with optimistic locking
    #      (_update_workflow_state(session_id, {'<column>': text}, expected_version))
    # The terminal trace UI can show each step: call trace.step_start('inventory_agent')
    # before the worker runs and trace.step_done('inventory_agent', old_version) after.

    # Implement route_to_inventory_agent
    @tool
    def route_to_inventory_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route an order-related request to the Inventory Agent to gather order facts.
        Call this FIRST for any request involving order status, history, or returns.

        Args:
            session_id:  The current session identifier (from the customer request)
            customer_id: The customer's unique identifier
            request:     The customer's original request

        Returns:
            Inventory facts retrieved by the InventoryAgent
        """
        workflow_state = _read_workflow_state(session_id) or {}
        old_version = int(workflow_state.get("version", 0))

        trace.step_start("inventory_agent")
        inventory_prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Request: {request}\n"
            "Use the inventory tools to fetch the relevant order and customer facts. "
            "Return the facts only, in JSON or structured text."
        )
        inventory_result = inventory_agent(inventory_prompt)
        inventory_text = str(inventory_result)

        _update_workflow_state(
            session_id,
            {"inventory_agent": inventory_text},
            old_version,
        )
        trace.step_done("inventory_agent", old_version)
        return inventory_text

    # Implement route_to_policy_agent
    @tool
    def route_to_policy_agent(session_id: str, request: str) -> str:
        """
        Route a policy question to the Policy Agent (multi-agent RAG).
        Call this for questions about return policies, shipping, or warranties.

        Args:
            session_id: The current session identifier
            request:    The customer's policy question

        Returns:
            Policy information retrieved and synthesized by PolicyAgent
        """
        workflow_state = _read_workflow_state(session_id) or {}
        old_version = int(workflow_state.get("version", 0))

        trace.step_start("policy_agent")
        policy_prompt = (
            f"Session ID: {session_id}\n"
            f"Request: {request}\n"
            "Use the policy retrieval tools to fetch the relevant return, shipping, "
            "or warranty policy details. Return the facts only, in JSON or structured text."
        )
        policy_result = policy_agent(policy_prompt)
        policy_text = str(policy_result)

        _update_workflow_state(
            session_id,
            {"policy_agent": policy_text},
            old_version,
        )
        trace.step_done("policy_agent", old_version)
        return policy_text

    # Implement route_to_refund_agent
    @tool
    def route_to_refund_agent(session_id: str, customer_id: str, request: str) -> str:
        """
        Route a return/refund request to the Refund Agent.
        Call this AFTER route_to_inventory_agent has gathered order facts.

        Args:
            session_id:  The current session identifier
            customer_id: The customer's unique identifier
            request:     The return/refund request

        Returns:
            Refund decision from the RefundAgent
        """
        workflow_state = _read_workflow_state(session_id) or {}
        old_version = int(workflow_state.get("version", 0))

        trace.step_start("refund_agent")
        refund_prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Request: {request}\n"
            "Use the inventory facts already stored in WorkflowState and the refund tools "
            "to determine refund eligibility. Return only the refund decision and any required details."
        )
        refund_result = refund_agent(refund_prompt)
        refund_text = str(refund_result)

        _update_workflow_state(
            session_id,
            {"refund_agent": refund_text},
            old_version,
        )
        trace.step_done("refund_agent", old_version)
        return refund_text

    # Implement route_to_communication_agent
    @tool
    def route_to_communication_agent(session_id: str, customer_id: str,
                                     original_request: str) -> str:
        """
        Route to the Communication Agent to compose the final customer response.
        Call this LAST - after all relevant worker agents have run.

        Args:
            session_id:       The current session identifier
            customer_id:      The customer's unique identifier
            original_request: The customer's original message

        Returns:
            Final customer-facing response drafted by CommunicationAgent
        """
        workflow_state = _read_workflow_state(session_id) or {}
        old_version = int(workflow_state.get("version", 0))

        trace.step_start("communication_agent")
        communication_prompt = (
            f"Session ID: {session_id}\n"
            f"Customer ID: {customer_id}\n"
            f"Original Request: {original_request}\n"
            "Read the full WorkflowState, include all previous agent findings, and compose a "
            "coherent, empathetic customer-facing response."
        )
        communication_result = communication_agent(communication_prompt)
        communication_text = str(communication_result)

        _update_workflow_state(
            session_id,
            {"communication_agent": communication_text},
            old_version,
        )
        trace.step_done("communication_agent", old_version)
        return communication_text

    # Implement initialize_session
    @tool
    def initialize_session(session_id: str, customer_id: str) -> str:
        """
        Create a blank WorkflowState record at the start of each new session.
        Call this at the VERY BEGINNING of processing every customer request.

        Args:
            session_id:  A unique identifier for this session
            customer_id: The customer's identifier

        Returns:
            Confirmation that the session was initialized
        """
        try:
            state = _create_workflow_state(session_id, customer_id)
            return json.dumps({
                "status": "initialized",
                "session_id": session_id,
                "customer_id": customer_id,
                "version": state.get("version", 0),
            }, indent=2)
        except ClientError as exc:
            current = _read_workflow_state(session_id) or {}
            if current:
                return json.dumps({
                    "status": "existing_session",
                    "session_id": session_id,
                    "customer_id": customer_id,
                    "version": current.get("version", 0),
                }, indent=2)
            raise RuntimeError(
                f"Failed to initialize workflow session {session_id}: {exc}"
            ) from exc

    # Instantiate and return the OrchestratorAgent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  AGENT GRAPH HELPERS
# ═══════════════════════════════════════════════════════

def _apply_guardrail(agents: list) -> None:
    """
    Attach the Bedrock Guardrail (Task 3) to every agent's BedrockModel.
    Guardrails are enforced per model invocation, so once GUARDRAIL_ID /
    GUARDRAIL_VERSION are known (in .env locally, as runtime environment
    variables when deployed) every agent in the graph runs behind the
    guardrail - no change to the agents themselves is needed.
    """
    guardrail_id      = config.GUARDRAIL_ID
    guardrail_version = config.GUARDRAIL_VERSION
    if not guardrail_id or not guardrail_version:
        return
    for agent in agents:
        model = getattr(agent, 'model', None)
        if model is not None and hasattr(model, 'update_config'):
            model.update_config(guardrail_id=guardrail_id,
                                guardrail_version=guardrail_version)


def build_agent_graph(verbose: bool = False) -> Agent:
    """Build all five agents, apply the guardrail, return the orchestrator."""
    def _ok(label):
        if verbose:
            print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  {label}{_C.RESET}", flush=True)

    inventory_agent     = build_inventory_agent();     _ok('InventoryAgent')
    refund_agent        = build_refund_agent();        _ok('RefundAgent')
    policy_agent        = build_policy_agent();        _ok('PolicyAgent')
    communication_agent = build_communication_agent(); _ok('CommunicationAgent')
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    _ok('Orchestrator')
    _apply_guardrail([inventory_agent, refund_agent, policy_agent,
                      communication_agent, orchestrator])
    if verbose and config.GUARDRAIL_ID:
        print(f"  {_C.GRY}          Guardrail {config.GUARDRAIL_ID} "
              f"(v{config.GUARDRAIL_VERSION}) attached to all agents{_C.RESET}")
    return orchestrator


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT TOOLING - AgentCore CLI
#
#  The runtime is deployed with the AgentCore CLI (`agentcore`, npm package
#  @aws/agentcore - https://github.com/aws/agentcore-cli) through the
#  pre-written helper src/agentcore_cli.py:
#
#    agentcore_cli.stage_runtime_code()   copies this file, its helper modules
#                                         and config.py to build/runtime/ with a
#                                         pyproject.toml of the runtime deps
#    agentcore_cli.configure_runtime()    writes the runtime settings (network
#                                         mode, protocol, execution role, env
#                                         vars) to agentcore/agentcore.json
#    agentcore_cli.deploy()               runs `agentcore deploy -y`: the CLI
#                                         downloads arm64 / Python 3.12 wheels
#                                         with uv, zips them with the code
#                                         (direct code deployment) and creates
#                                         or updates the runtime via CDK
#    agentcore_cli.deployed_runtime_arn() reads the ARN the CLI recorded
#
#  Inside the runtime this same file is the entry point: it is started with
#  no command-line argument and serves HTTP (see run_serve). The marker file
#  written next to it by stage_runtime_code() tells __main__ to do so.
# ═══════════════════════════════════════════════════════

_RUNTIME_MARKER = '.agentcore-runtime'         # written by agentcore_cli.stage_runtime_code()
_SRC_DIR        = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════

def create_guardrail() -> tuple[str, str]:
    """
    Create a Bedrock Guardrail for enterprise safety enforcement.

    Blocks harmful content, PII exposure, off-topic subjects, and profanity.
    Returns (guardrail_id, guardrail_version).
    """
    bedrock_client = boto3.client('bedrock', region_name=config.AWS_REGION)
    
    # Check if guardrail already exists to avoid duplicates
    existing = bedrock_client.list_guardrails()
    for g in existing.get('guardrails', []):
        if g['name'] == config.GUARDRAIL_NAME:
            guardrail_id = g['id']
            versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
            guardrail_version = 'DRAFT'
            for v in versions.get('guardrails', []):
                if v.get('version', 'DRAFT') != 'DRAFT':
                    guardrail_version = v['version']
            print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
            return guardrail_id, guardrail_version
    
    # existing = bedrock_client.list_guardrails()
    # for g in existing.get('guardrails', []):
    #     if g['name'] == config.GUARDRAIL_NAME:
    #         guardrail_id = g.get('id', g.get('guardrailId'))
    #         versions = bedrock_client.list_guardrails(guardrailIdentifier=guardrail_id)
    #         guardrail_version = None
    #         for v in versions.get('guardrails', []):
    #             version = v.get('version', v.get('guardrailVersion', 'DRAFT'))
    #             if version != 'DRAFT':
    #                 guardrail_version = version
    #         if guardrail_version is None:
    #             promoted = bedrock_client.create_guardrail_version(
    #                 guardrailIdentifier=guardrail_id,
    #                 description='Promote the NovaMart enterprise safety guardrail.',
    #             )
    #             guardrail_version = promoted['guardrailVersion']
    #         print(f"Guardrail already exists: {guardrail_id} (version: {guardrail_version})")
    #         return guardrail_id, guardrail_version

    topic_definitions = []
    for topic in config.GUARDRAIL_BLOCKED_TOPICS:
        if isinstance(topic, dict):
            topic_name = str(
                topic.get('name') or topic.get('topic') or topic.get('topicName') or ''
            )
        else:
            topic_name = str(topic)
        normalized = topic_name.lower().replace('_', ' ').replace('-', ' ')

        if 'compet' in normalized:
            definition = (
                'Requests to identify, recommend, compare, promote, or provide product '
                'details or purchasing advice for products sold by competitors instead '
                'of NovaMart. This topic is limited to competitor products.'
            )
        elif any(word in normalized for word in ('price', 'haggl', 'negotiat')):
            definition = (
                'Requests to haggle over, negotiate, price-match, lower, or otherwise '
                'change an advertised or quoted NovaMart price, including requests for '
                'a special price exception. Do not classify arithmetic using an already-'
                'specified price and discount, such as calculating the discounted total '
                'or savings, as price negotiation.'
            )
        elif any(word in normalized for word in ('legal', 'threat', 'litigat')):
            definition = (
                'Requests to threaten or pursue legal action against NovaMart, or to '
                'draft or send a legal threat, demand, or litigation notice. This topic '
                'is limited to legal threats and escalation.'
            )
        else:
            definition = f'Requests concerning {topic_name}, as identified by the topic name.'

        topic_definitions.append({
            'name': topic_name[:100],
            'definition': definition,
            'type': 'DENY',
        })

    response = bedrock_client.create_guardrail(
        name=config.GUARDRAIL_NAME,
        description='Enterprise safety policies for NovaMart customer support agents.',
        contentPolicyConfig={
            'filtersConfig': [
                {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                {'type': 'INSULTS', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
                {'type': 'MISCONDUCT', 'inputStrength': 'MEDIUM', 'outputStrength': 'MEDIUM'},
            ]
        },
        sensitiveInformationPolicyConfig={
            'piiEntitiesConfig': [
                {'type': 'CREDIT_DEBIT_CARD_NUMBER', 'action': 'BLOCK'},
                {'type': 'US_SOCIAL_SECURITY_NUMBER', 'action': 'BLOCK'},
                {'type': 'EMAIL', 'action': 'ANONYMIZE'},
                {'type': 'PHONE', 'action': 'ANONYMIZE'},
            ]
        },
        topicPolicyConfig={
            'topicsConfig': topic_definitions,
            'tierConfig': {'tierName': 'STANDARD'},
        },
        wordPolicyConfig={
            'managedWordListsConfig': [{'type': 'PROFANITY'}]
        },
        blockedInputMessaging='Your request was blocked by the NovaMart safety policy.',
        blockedOutputsMessaging='The response was blocked by the NovaMart safety policy.',
        crossRegionConfig={'guardrailProfileIdentifier': 'us.guardrail.v1:0'},
    )

    guardrail_id = response['guardrailId']
    promoted = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description='Versioned NovaMart enterprise safety guardrail.',
    )
    guardrail_version = promoted['guardrailVersion']
    print(f"Created guardrail: {guardrail_id} (version: {guardrail_version})")
    return guardrail_id, guardrail_version



def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str,
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime
    using the AgentCore CLI.

    The CLI packages the staged application and deploys it through
    CloudFormation. Guardrail configuration is passed through runtime
    environment variables and applied to each agent's model by
    _apply_guardrail().

    Returns:
        The deployed AgentCore Runtime ARN.
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME

    print(
        f"  AWS Account: {config.ACCOUNT_ID}  "
        f"| Region: {config.AWS_REGION}"
    )
    print(f"  Runtime: {runtime_name}")

    # Stage the application files that the CLI will package.
    agentcore_cli.stage_runtime_code()
    # Build the runtime environment variables.
    runtime_env = {
        "AWS_REGION": config.AWS_REGION,
        "PROJECT_NAME": config.PROJECT_NAME,
        "RETURNS_KB_ID": config.RETURNS_KB_ID,
        "SHIPPING_KB_ID": config.SHIPPING_KB_ID,
        "WARRANTY_KB_ID": config.WARRANTY_KB_ID,
        "AGENT_LOG_GROUP": config.AGENT_LOG_GROUP,
        "GUARDRAIL_ID": guardrail_id,
        "GUARDRAIL_VERSION": guardrail_version,
    }

    # Configure the runtime through the AgentCore CLI.
    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode="PUBLIC",
        protocol="HTTP",
        execution_role_arn=config.AGENTCORE_ROLE_ARN,
    )

    # Package and deploy the application through CloudFormation.
    agentcore_cli.deploy()

    # Retrieve and return the deployed runtime ARN.
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError(
            "deploy_to_agentcore_runtime: "
            "AgentCore runtime ARN not found after deploy"
        )

    print(f"  Runtime deployed: {runtime_arn}")
    return runtime_arn


# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Create an AgentCore Memory resource for session-scoped conversational
    context. Uses the SESSION_SUMMARY (summaryMemoryStrategy) strategy with
    7-day event retention.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAME
    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    response = agentcore_control.create_memory(
        name=memory_name,
        description='Session summary memory for NovaMart customer support conversations.',
        eventExpiryDuration=7,
        memoryStrategies=[{
            'summaryMemoryStrategy': {
                'name': 'SessionSummary',
                'namespaces': ['/summaries/{actorId}/{sessionId}'],
            }
        }],
        clientToken=str(uuid.uuid4()),
    )

    memory = response['memory']
    print(f"  Memory created: {memory['arn']}  (status: {memory['status']})")
    print("  Waiting for memory status ACTIVE", end='', flush=True)
    deadline = time.time() + 300
    while memory['status'] != 'ACTIVE' and time.time() < deadline:
        time.sleep(10)
        print('.', end='', flush=True)
        memory = agentcore_control.get_memory(memoryId=memory['id'])['memory']
        if memory['status'] == 'FAILED':
            raise RuntimeError(f"Memory creation failed: {memory.get('failureReason')}")
    print(' ready.' if memory['status'] == 'ACTIVE' else f" status {memory['status']}")
    return memory['arn']


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════

def configure_observability(runtime_arn: str) -> None:
    """
    Configure observability for the deployed agent:
    - Agent logs → CloudWatch Logs at INFO level (config.AGENT_LOG_GROUP)
    - Execution traces → AWS X-Ray at 100% sampling

    The loggingConfiguration built here is applied by
    apply_observability_config() (agent_observability.py):
      cloudWatchConfig -> log group created; runtime env AGENT_LOG_GROUP /
                          AGENT_LOG_LEVEL so the deployed agent ships its logs there
      xRayConfig       -> CloudWatch Transaction Search enabled with the given
                          sampling percentage; runtime env AGENT_TRACING_ENABLED /
                          AGENT_TRACE_SAMPLING_RATE
    """
    # Build the logging configuration
    # logging_configuration = {
    #     'cloudWatchConfig': {'logGroupName': config.AGENT_LOG_GROUP,
    #                          'logLevel': 'INFO', 'enabled': True},
    #     'xRayConfig':       {'enabled': True, 'samplingRate': 1.0},
    # }
    # Then apply it:  summary = apply_observability_config(runtime_arn, logging_configuration)
    # Wrap the call in try/except - on success print the CloudWatch log group
    # and the X-Ray sampling rate; on exception print
    #   "[Note] Observability configuration failed: <e>"
    logging_configuration = {
        'cloudWatchConfig': {
            'logGroupName': config.AGENT_LOG_GROUP,
            'logLevel': 'INFO',
            'enabled': True
        },
        'xRayConfig': {
            'enabled': True,
            'samplingRate': 1.0
        }
    }
    
    summary = None
    try:
        summary = apply_observability_config(runtime_arn, logging_configuration)
        print(f"  Observability configured: CloudWatch log group {summary['cloudWatchConfig']['logGroupName']}, "
              f"X-Ray sampling rate {summary['xRayConfig']['samplingRate']}")
    
    except Exception as e:
        print(f"[Note] Observability configuration failed: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Deployment pattern:
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION = os.environ.get('ORDERS_FUNCTION', '')
_POLICY_FUNCTION = os.environ.get('POLICY_FUNCTION', '')
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', '')


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Optional extension to the in-process @tool functions. Resolve configured
    Lambda functions first; if none exist, skip gateway creation. Otherwise
    create/reuse the gateway and submit its targets. Connecting agents to this
    MCP endpoint requires separate integration; this starter uses in-process tools.

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        A SKIPPED result with a reason, or gateway details and target count.
    """
    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    # Resolve optional Lambda targets before creating any gateway resources.
    available = []
    for target in targets:
        if not target['function']:
            continue
        try:
            available.append((target, _gw_get_function_arn(target['function'])))
        except ClientError as exc:
            if exc.response['Error']['Code'] != 'ResourceNotFoundException':
                raise
            print(f"    [Skipped] {target['name']}: Lambda function not found")

    if not available:
        return {'status': 'SKIPPED', 'reason': 'No configured Lambda tool functions are available.'}

    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    print(f"\n  Registering {len(available)} Gateway targets...")
    for target, lambda_arn in available:
        _gw_create_target(agentcore_ctrl, gateway_id, target, lambda_arn)

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url,
            'status': 'TARGETS_SUBMITTED', 'target_count': len(available)}



# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> dict:
    """
    Invoke the deployed agent via AgentCore Runtime (see run_serve).

    AgentCore requires runtimeSessionId to be at least 33 characters, so the
    short project session id is embedded in a longer, unique runtime session id.
    """
    if not config.AGENTCORE_RUNTIME_ARN:
        raise RuntimeError("AGENTCORE_RUNTIME_ARN is not set - run the deploy command first")

    runtime_session_id = f"{session_id}-{uuid.uuid4().hex}"     # >= 33 chars
    payload = json.dumps({
        'prompt':      user_message,
        'session_id':  session_id,
        'customer_id': customer_id,
    })
    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        runtimeSessionId=runtime_session_id,
        contentType='application/json',
        accept='application/json',
        payload=payload,
    )
    body = response['response'].read()
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        return {'result': body.decode('utf-8', errors='replace') if isinstance(body, bytes) else str(body)}


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    # Fail fast if the AgentCore CLI (used by Steps 3 and 5) is missing.
    import agentcore_cli
    print(f"AgentCore CLI: {agentcore_cli.cli_version()} ({agentcore_cli.cli_path()})\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        if gw['status'] == 'SKIPPED':
            print(f"  [Skipped] Gateway: {gw['reason']}")
        else:
            print(f"  Gateway URL : {gw['gateway_url']}")
            print("  Lambda targets submitted; connect an MCP client separately to use them.")
    except Exception as e:
        print(f"  [Note] Optional Gateway deployment failed: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    print(f"  Then try the deployed runtime:")
    print(f"  python src/agent_orchestrator.py invoke \"What is the return policy for premium customers?\"")
    print(f"  or with the CLI:  agentcore invoke \"What is the return policy for premium customers?\"")
    print(f"  (agentcore status / agentcore logs show the deployed runtime and its logs)\n")
    return runtime_arn, guardrail_id


# ═══════════════════════════════════════════════════════
#  LOCAL TEST SCENARIOS
# ═══════════════════════════════════════════════════════

# Order IDs match infrastructure/seed_data.py.
TEST_CASES = [
    ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
    ("CUST-002", "What is the return policy for premium customers?"),
    ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
]

# Test customers shown by the chat command. Data matches seed_data.py.
TEST_CUSTOMERS = [
    ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Wireless Headphones Pro"),
    ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "Mechanical Keyboard K2"),
    ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "Laptop UltraBook 14"),
    ("CUST-004", "David Lee",     "Standard", "ORD-30001", "Phone Case Slim"),
]


def run_test_scenarios() -> None:
    """Run the three scenarios locally; every request is traced to X-Ray."""
    print("Running local agent test...")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph()

    for customer_id, query in TEST_CASES:
        session_id = str(uuid.uuid4())[:8]
        print(f"\n{'─'*60}")
        print(f"Session: {session_id} | Customer: {customer_id}")
        print(f"Query: {query}")
        prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
        with tracer.trace_request(session_id, customer_id, query):
            response = orchestrator(prompt)
        print(f"Response: {response}")
        print_trace_hint()
    flush_logs()


def run_chat() -> None:
    """Interactive terminal chat - educational mode."""
    W = _C.W

    # ── Welcome banner ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
    print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
    print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

    # ── Test customers ────────────────────────────────────────────────
    print()
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.BOLD}Test Customers{_C.RESET}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
    print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
    for cid, name, tier, order, product in TEST_CUSTOMERS:
        tier_col = _C.INV if tier == 'Premium' else _C.GRY
        print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
              f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
    print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
    print()

    customer_id = (
        input(f"  Enter Customer ID (default: CUST-001): ").strip()
        or "CUST-001"
    )
    session_id  = str(uuid.uuid4())[:8]
    print()
    print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
    print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
    print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
    print()

    # ── Build agents and show initialization order.
    print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
    setup_logging(to_cloudwatch=True)
    orchestrator = build_agent_graph(verbose=True)
    print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
    print()

    # ── Conversation loop ─────────────────────────────────────────────
    while True:
        try:
            user_input = input(
                f"  {_C.BOLD}You >{_C.RESET} "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ('quit', 'exit', 'q'):
            print(f"  {_C.GRY}Session ended.{_C.RESET}")
            break

        prompt  = (f"[Session ID: {session_id}] "
                   f"[Customer ID: {customer_id}] {user_input}")
        t0_turn = time.time()

        # ── Install proxy, run orchestrator (traced), restore stdout ───
        trace.new_turn()
        sys.stdout = _trace_writer
        try:
            with tracer.trace_request(session_id, customer_id, user_input):
                response = orchestrator(prompt)
        finally:
            sys.stdout = _real_stdout   # always restore, even on exception

        elapsed = time.time() - t0_turn

        # ── Resolve the final customer-facing text ────────────────────
        final_state = _read_workflow_state(session_id) or {}
        comm_result = final_state.get('communication_agent', '')
        text = _strip_xml_tags(comm_result or str(response))

        # ── DynamoDB workflow state summary ───────────────────────────
        trace.summary(session_id, elapsed)

        # ── Final customer-facing response ────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        for line in text.splitlines():
            print(f"  {line}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        if tracer.last_trace_id:
            print(f"  {_C.GRY}X-Ray trace : {tracer.last_trace_id}"
                  f"{'' if tracer.last_published else '  (not published)'}{_C.RESET}")
        print()
    flush_logs()


def run_invoke(message: str, customer_id: str = "CUST-001") -> None:
    """Send one message to the deployed AgentCore Runtime and print the reply."""
    session_id = str(uuid.uuid4())[:8]
    print(f"Invoking {config.AGENTCORE_RUNTIME_ARN}")
    print(f"Session: {session_id} | Customer: {customer_id}")
    print(f"Query: {message}\n")
    result = invoke_agent(session_id, customer_id, message)
    print(f"Response: {result.get('result', result)}")
    if result.get('trace_id'):
        print(f"X-Ray trace: {result['trace_id']}")


def run_serve() -> None:
    """
    HTTP entry point executed inside Amazon Bedrock AgentCore Runtime.

    BedrockAgentCoreApp (bedrock-agentcore SDK) exposes the contract the
    runtime expects - POST /invocations and GET /ping on port 8080 - and hands
    each request payload to the function decorated with @app.entrypoint.

    Request payload (see invoke_agent):
        {"prompt": "<customer message>", "customer_id": "CUST-001", "session_id": "abc12345"}
    Response:
        {"result": "<final customer-facing text>", "session_id": ..., "trace_id": ...}

    The five-agent graph is built once (first request) and reused. Guardrail,
    tracing and logging are applied exactly as in the local test/chat modes,
    from the runtime's environment variables.
    """
    from bedrock_agentcore import BedrockAgentCoreApp

    os.environ.setdefault('AGENT_RUNTIME_MODE', 'agentcore-runtime')
    if os.environ.get('AGENT_LOG_GROUP') and 'AGENT_LOG_TO_CLOUDWATCH' not in os.environ:
        os.environ['AGENT_LOG_TO_CLOUDWATCH'] = 'true'

    app   = BedrockAgentCoreApp()
    lock  = threading.Lock()
    graph = {}

    def _orchestrator():
        with lock:
            if 'agent' not in graph:
                setup_logging()
                graph['agent'] = build_agent_graph()
        return graph['agent']

    @app.entrypoint
    def invoke(payload, context=None):
        payload     = payload or {}
        prompt      = payload.get('prompt') or payload.get('message') or ''
        customer_id = payload.get('customer_id') or 'CUST-001'
        session_id  = payload.get('session_id') or (
            getattr(context, 'session_id', None) or uuid.uuid4().hex)[:8]
        if not prompt:
            return {'error': "payload must include 'prompt'"}

        enriched = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {prompt}"
        with tracer.trace_request(session_id, customer_id, prompt):
            response = _orchestrator()(enriched)

        state = _read_workflow_state(session_id) or {}
        text  = _strip_xml_tags(state.get('communication_agent', '') or str(response))
        flush_logs()
        return {'result': text, 'session_id': session_id, 'customer_id': customer_id,
                'trace_id': tracer.last_trace_id}

    app.run()


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''

    # Inside the AgentCore Runtime package (marker file next to this script)
    # the entry point is started without arguments -> serve HTTP.
    if not command and os.path.exists(os.path.join(_SRC_DIR, _RUNTIME_MARKER)):
        command = 'serve'

    if command == 'deploy':
        deploy_all()

    elif command == 'serve':
        run_serve()

    elif command == 'test':
        run_test_scenarios()

    elif command == 'chat':
        run_chat()

    elif command == 'invoke':
        if len(sys.argv) < 3:
            print('Usage: python src/agent_orchestrator.py invoke "<message>" [CUSTOMER_ID]')
            sys.exit(1)
        run_invoke(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "CUST-001")

    else:
        print("Usage:")
        print("  python src/agent_orchestrator.py deploy           # Deploy to AgentCore (Tasks 3-6)")
        print("  python src/agent_orchestrator.py test             # Run the 3 test scenarios locally")
        print("  python src/agent_orchestrator.py chat             # Interactive terminal chat")
        print("  python src/agent_orchestrator.py invoke \"<msg>\"   # Call the deployed runtime")
        print("  python src/agent_orchestrator.py serve            # HTTP server (used inside AgentCore Runtime)")
