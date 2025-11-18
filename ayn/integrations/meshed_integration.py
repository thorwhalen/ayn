"""Integration with meshed package for agent pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..controllers.base import BaseAgentController

# meshed integration is optional
try:
    from meshed import DAG
    HAS_MESHED = True
except ImportError:
    HAS_MESHED = False


def pipeline(*agents: BaseAgentController, names: Optional[List[str]] = None) -> Callable:
    """Create a sequential pipeline of agents using meshed.

    Args:
        *agents: Agent controllers to chain
        names: Optional names for each agent in the pipeline

    Returns:
        Callable pipeline function

    Example:
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> from ayn import GenericController
        >>>
        >>> # Create test agents
        >>> meta1 = AgentMetadata(name="agent1", description="test", framework=AgentFramework.CUSTOM)
        >>> meta2 = AgentMetadata(name="agent2", description="test", framework=AgentFramework.CUSTOM)
        >>> agent1 = GenericController(meta1, ControllerConfig())
        >>> agent2 = GenericController(meta2, ControllerConfig())
        >>>
        >>> # Without meshed, creates simple sequential pipeline
        >>> pipe = pipeline(agent1, agent2)
        >>> # result = pipe({"input": "test"})
    """
    if not HAS_MESHED:
        # Fallback: simple sequential execution
        def simple_pipeline(input_data: Any) -> Any:
            result = input_data
            for agent in agents:
                result = agent.invoke(result)
            return result

        return simple_pipeline

    # Use meshed for more sophisticated DAG execution
    names = names or [f"agent_{i}" for i in range(len(agents))]

    # Create functions for each agent
    funcs = {}
    for i, (agent, name) in enumerate(zip(agents, names)):
        if i == 0:
            # First agent takes input directly
            funcs[name] = agent.invoke
        else:
            # Subsequent agents take previous output
            prev_name = names[i - 1]
            funcs[name] = lambda data, a=agent: a.invoke(data)

    # Create DAG
    dag = DAG(funcs)

    return dag


def create_agent_dag(
    agents: Dict[str, BaseAgentController],
    dependencies: Dict[str, List[str]],
) -> Callable:
    """Create a DAG of agents with explicit dependencies.

    Args:
        agents: Dict mapping agent names to controllers
        dependencies: Dict mapping agent names to list of dependency names

    Returns:
        Callable DAG function

    Example:
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> from ayn import GenericController
        >>>
        >>> meta1 = AgentMetadata(name="extract", description="test", framework=AgentFramework.CUSTOM)
        >>> meta2 = AgentMetadata(name="transform", description="test", framework=AgentFramework.CUSTOM)
        >>> meta3 = AgentMetadata(name="load", description="test", framework=AgentFramework.CUSTOM)
        >>>
        >>> agents = {
        ...     "extract": GenericController(meta1, ControllerConfig()),
        ...     "transform": GenericController(meta2, ControllerConfig()),
        ...     "load": GenericController(meta3, ControllerConfig()),
        ... }
        >>>
        >>> dependencies = {
        ...     "transform": ["extract"],
        ...     "load": ["transform"],
        ... }
        >>>
        >>> # Without meshed, creates simple execution order
        >>> dag = create_agent_dag(agents, dependencies)
        >>> # result = dag({"data": "test"})
    """
    if not HAS_MESHED:
        # Fallback: topological sort and sequential execution
        def simple_dag(input_data: Any) -> Dict[str, Any]:
            results = {}
            executed = set()

            def execute(agent_name: str):
                if agent_name in executed:
                    return results[agent_name]

                # Execute dependencies first
                deps = dependencies.get(agent_name, [])
                dep_results = {}
                for dep in deps:
                    if dep not in executed:
                        execute(dep)
                    dep_results[dep] = results[dep]

                # Execute this agent
                if deps:
                    # If has dependencies, pass their results
                    agent_input = dep_results
                else:
                    # If no dependencies, use original input
                    agent_input = input_data

                result = agents[agent_name].invoke(agent_input)
                results[agent_name] = result
                executed.add(agent_name)
                return result

            # Execute all agents
            for agent_name in agents:
                execute(agent_name)

            return results

        return simple_dag

    # Use meshed for proper DAG execution
    funcs = {}

    for agent_name, agent in agents.items():
        deps = dependencies.get(agent_name, [])

        if not deps:
            # No dependencies - takes input directly
            funcs[agent_name] = agent.invoke
        else:
            # Has dependencies - takes their outputs
            def make_func(a, d):
                def func(**kwargs):
                    # Collect dependency results
                    dep_results = {k: v for k, v in kwargs.items() if k in d}
                    return a.invoke(dep_results)

                return func

            funcs[agent_name] = make_func(agent, deps)

    # Create DAG with dependencies
    dag = DAG(funcs)

    return dag


class AgentPipeline:
    """Higher-level pipeline builder for agents.

    Example:
        >>> from ayn.core import AgentMetadata, AgentFramework, ControllerConfig
        >>> from ayn import GenericController
        >>>
        >>> meta1 = AgentMetadata(name="agent1", description="test", framework=AgentFramework.CUSTOM)
        >>> agent1 = GenericController(meta1, ControllerConfig())
        >>>
        >>> pipeline_builder = AgentPipeline()
        >>> pipeline_builder.add_stage("process", agent1)
        >>> # result = pipeline_builder.run({"input": "test"})
    """

    def __init__(self):
        self.stages: List[tuple[str, BaseAgentController]] = []

    def add_stage(self, name: str, agent: BaseAgentController):
        """Add a stage to the pipeline.

        Args:
            name: Name of the stage
            agent: Agent controller for this stage
        """
        self.stages.append((name, agent))
        return self

    def run(self, input_data: Any) -> Dict[str, Any]:
        """Run the pipeline.

        Args:
            input_data: Initial input

        Returns:
            Dict mapping stage names to outputs
        """
        results = {}
        current_data = input_data

        for name, agent in self.stages:
            output = agent.invoke(current_data)
            results[name] = output
            current_data = output

        return results

    def parallel_stage(self, agents: Dict[str, BaseAgentController]):
        """Add a parallel stage with multiple agents.

        Args:
            agents: Dict mapping names to agents

        Returns:
            Self for chaining
        """
        # Store as a special parallel marker
        self.stages.append(("__parallel__", agents))
        return self

    def run_with_parallel(self, input_data: Any) -> Dict[str, Any]:
        """Run pipeline with parallel stages.

        Args:
            input_data: Initial input

        Returns:
            Results from all stages
        """
        results = {}
        current_data = input_data

        for name, agent in self.stages:
            if name == "__parallel__":
                # Run all agents in parallel with same input
                parallel_results = {}
                for agent_name, agent_instance in agent.items():
                    parallel_results[agent_name] = agent_instance.invoke(current_data)
                results.update(parallel_results)
                current_data = parallel_results
            else:
                output = agent.invoke(current_data)
                results[name] = output
                current_data = output

        return results
