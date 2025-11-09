# Graph Structure Explanation

## Overview

The validation agent uses a **two-level graph structure**:

1. **Outer Graph**: Simple linear flow (START → validate_cpt_icd_codes → END)
2. **Inner ReAct Agent Graph**: Dynamic reasoning loop with tool calls

## Outer Graph Structure

```
START → validate_cpt_icd_codes → END
```

This is a simple StateGraph with:
- **Input**: `GraphState` (claim_data, discharge_summary, provider_email, agent_messages, validation_results)
- **Output**: `GraphOutput` (claim_valid, justification)
- **Single Node**: `validate_cpt_icd_codes` - Contains the ReAct agent logic

## Inner ReAct Agent Graph

The `validate_cpt_icd_codes` node contains a **ReAct agent** created by `create_react_agent()`. This agent has its own internal graph structure:

```
Agent Start
  ↓
Should Continue? (Decision Node)
  ↓ (if yes)
Action Node (LLM decides which tool to call)
  ↓
Tool Execution (one of):
  - cpt_lookup(code)
  - icd_lookup(code)
  - send_email(email_data)
  ↓
Observation (Tool results)
  ↓
Should Continue? (Loop back or finish)
  ↓ (if done)
Final Answer
```

## Dynamic Flow

The ReAct agent uses **dynamic edges** based on:
1. **LLM reasoning**: Decides which tool to call next
2. **Tool results**: Observations feed back into the reasoning loop
3. **Completion condition**: Agent decides when validation is complete

## Tools Available to Agent

1. **cpt_lookup(code)**: Searches CPT code index using `sdk.context_grounding.search_async()`
2. **icd_lookup(code)**: Searches ICD code index using `sdk.context_grounding.search_async()`
3. **send_email(email_data)**: Invokes UiPath process using `sdk.processes.invoke_async()`

## State Flow

### Outer Graph State
- `GraphState` → Contains claim data, summary, email, tracing fields
- Flows through single node and returns `GraphOutput`

### Inner Agent State
- `{"messages": [...]}` → LangChain message history
- Contains system prompt, user input, tool calls, and observations
- Agent maintains conversation history for reasoning

## Key Features

1. **Agent Reasoning**: All validation logic is done by agent reasoning, not hardcoded
2. **Dynamic Tool Selection**: Agent chooses which tools to call and when
3. **Tracing Support**: Logging at each step for debugging
4. **HTML Email Generation**: Agent prepares HTML emails in reasoning, not as tool input

## Visualization

See `validate-coding-agent.mermaid` for a visual diagram of the graph structure.

