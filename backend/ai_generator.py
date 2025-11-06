from typing import List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to tools for searching course information and retrieving course outlines.

Tool Usage Guidelines:
- **Course outline queries** (syllabus, lesson list, course structure): Use the `get_course_outline` tool
  - Returns: course title, course link, instructor, and complete lesson list with numbers and titles
  - Present all lessons in a clear, numbered format
- **Content search queries** (specific topics, concepts, lesson details): Use the `search_course_content` tool
  - Returns: relevant course content matching the query
  - Synthesize search results into accurate, fact-based responses
- **Multi-step reasoning**: You may use up to 2 tool calls per query if needed
  - Example: Search for content, then retrieve outline for additional context
  - Example: Search one course, then search another for comparison
  - Only use multiple tools when genuinely beneficial to answering the query
- **Tool efficiency**: Prefer single tool calls when sufficient; use multiple calls only when the first result is incomplete
- If tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline requests**: Use the outline tool, then present the complete course structure
- **Course content questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"
 - Do not narrate your tool-calling strategy ("First I'll search X, then I'll check Y")

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with up to 2 sequential tool calls.

        Termination conditions:
        1. Claude returns stop_reason != "tool_use" (natural completion)
        2. Round count reaches MAX_TOOL_ROUNDS (from config)
        3. Tool execution error occurs

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """
        from config import config

        # Build system content with conversation history
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize message history with user query
        messages = [{"role": "user", "content": query}]

        # Tool execution loop
        round_count = 0
        MAX_ROUNDS = config.MAX_TOOL_ROUNDS

        while round_count < MAX_ROUNDS:
            # Prepare API parameters with tools preserved
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Include tools if provided and tool_manager available
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Make API call
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                return f"Error generating response: {str(e)}"

            # Check termination conditions
            if response.stop_reason != "tool_use":
                # Claude chose to respond directly - we're done
                return self._extract_text(response)

            # Tool execution path
            if not tool_manager:
                # No tool manager available despite tool_use
                return self._extract_text(response)

            # Increment round counter
            round_count += 1

            # Add assistant's tool use response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Execute all tool calls and collect results
            tool_results = []
            for content_block in response.content:
                if content_block.type == "tool_use":
                    try:
                        # Execute the tool
                        tool_result = tool_manager.execute_tool(
                            content_block.name, **content_block.input
                        )

                        # Build tool_result message block
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result,
                            }
                        )

                    except Exception as e:
                        # Tool execution failed - add error and attempt final response
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": f"Error executing tool: {str(e)}",
                                "is_error": True,
                            }
                        )

            # Add tool results as user message
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        # If we exit loop due to MAX_ROUNDS, make final call WITHOUT tools
        # This forces Claude to synthesize final answer from accumulated context
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Note: NO tools parameter - forces synthesis
        }

        try:
            final_response = self.client.messages.create(**final_params)
            return self._extract_text(final_response)
        except Exception as e:
            return f"Error generating final response: {str(e)}"

    def _extract_text(self, response) -> str:
        """
        Extract text content from Claude API response.

        Args:
            response: Claude API response object

        Returns:
            Extracted text content
        """
        for content_block in response.content:
            if hasattr(content_block, "text"):
                return content_block.text
        return "No response generated"
