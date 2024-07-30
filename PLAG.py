# Graph-enhanced Large Language Models in Asynchronous Plan Reasoning
# https://x.com/FangruLin99/status/1797979103632699459


import openai
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

class PLaGPromptFramework:
    def __init__(self, model: str = "gpt-4", api_key: str = None):
        self.model = model
        if api_key:
            openai.api_key = api_key

    def call_llm(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return ""

    def extract_steps(self, text: str) -> List[Tuple[str, int]]:
        prompt = f"""
        Extract the steps and their durations from the following text. 
        Format the output as a list of tuples: (step, duration_in_minutes).

        Text: {text}
        """
        response = self.call_llm(prompt)
        try:
            return eval(response)
        except Exception as e:
            print(f"Error parsing LLM response for steps: {e}")
            return []

    def generate_graph(self, steps: List[Tuple[str, int]]) -> nx.DiGraph:
        G = nx.DiGraph()
        for i, (step, duration) in enumerate(steps):
            G.add_node(i, label=step, duration=duration)
            if i > 0:
                G.add_edge(i-1, i)
        return G

    def visualize_graph(self, G: nx.DiGraph, title: str = ""):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=8, font_weight='bold')
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=6)
        plt.title(title)
        plt.show()

    def refine_graph(self, G: nx.DiGraph) -> nx.DiGraph:
        for node in G.nodes:
            llm_prompt = f"Refine the details of the task '{G.nodes[node]['label']}' in the context of the overall plan. Provide additional information or considerations."
            response = self.call_llm(llm_prompt)
            G.nodes[node]['details'] = response
        for edge in G.edges:
            llm_prompt = f"Explain the relationship or dependencies between the tasks '{G.nodes[edge[0]]['label']}' and '{G.nodes[edge[1]]['label']}' in the context of the overall plan."
            response = self.call_llm(llm_prompt)
            G.edges[edge]['details'] = response
        return G

    def optimize_plan(self, G: nx.DiGraph) -> Dict[str, Any]:
        prompt = f"""
        Given the following graph representation of a plan:

        Nodes: {dict(G.nodes(data=True))}
        Edges: {dict(G.edges(data=True))}

        Optimize this plan to minimize the total duration. Consider parallel execution where possible.
        Return the optimized plan as a dictionary with the following structure:
        {{"total_duration": int, "execution_order": List[List[int]]}}

        The execution_order should be a list of lists, where each inner list represents steps that can be executed in parallel.
        """
        response = self.call_llm(prompt)
        try:
            return eval(response)
        except Exception as e:
            print(f"Error parsing LLM response for optimized plan: {e}")
            return {"total_duration": 0, "execution_order": []}

    def graph_to_prompt(self, G: nx.DiGraph) -> str:
        prompt = "The refined task breakdown is as follows:\n\n"
        for node in G.nodes:
            prompt += f"Task: {G.nodes[node]['label']}\n"
            prompt += f"Duration: {G.nodes[node]['duration']} minutes\n"
            prompt += f"Details: {G.nodes[node].get('details', 'N/A')}\n\n"
        prompt += "Task Dependencies:\n"
        for edge in G.edges:
            prompt += f"{G.nodes[edge[0]]['label']} -> {G.nodes[edge[1]]['label']}\n"
            prompt += f"Details: {G.edges[edge].get('details', 'N/A')}\n\n"
        return prompt

    def refine_prompt(self, initial_prompt: str) -> str:
        steps = self.extract_steps(initial_prompt)
        if not steps:
            return "Error: Unable to extract steps from the initial prompt."

        G = self.generate_graph(steps)
        self.visualize_graph(G, "Initial Graph")

        G = self.refine_graph(G)
        self.visualize_graph(G, "Refined Graph")

        optimized_plan = self.optimize_plan(G)
        if not optimized_plan["execution_order"]:
            return "Error: Unable to optimize the plan."

        refined_prompt = f"""
        Original task: {initial_prompt}

        Refined task breakdown:
        {self.graph_to_prompt(G)}

        Optimized plan:
        Total duration: {optimized_plan['total_duration']} minutes
        Execution order: {optimized_plan['execution_order']}

        Please provide a detailed step-by-step guide on how to execute this optimized plan, explaining:
        1. The order of tasks
        2. Which tasks can be done in parallel
        3. How to coordinate between parallel tasks
        4. Any specific considerations or details for each task
        5. Tips for efficient execution of the overall plan
        """
        return refined_prompt

# Usage
framework = PLaGPromptFramework(api_key="your_openai_api_key")
initial_prompt = """
I need to prepare for a dinner party. The tasks are:
1. Clean the house (60 minutes)
2. Go grocery shopping (45 minutes)
3. Prepare the appetizers (30 minutes)
4. Cook the main course (90 minutes)
5. Set the table (15 minutes)
6. Get dressed (20 minutes)

What's the most efficient way to do all of this?
"""

refined_prompt = framework.refine_prompt(initial_prompt)
print("Refined prompt:")
print(refined_prompt)

final_response = framework.call_llm(refined_prompt)
print("\nFinal response:")
print(final_response)
