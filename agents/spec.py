from typing import Callable, List

class AgentSpec:
    def __init__(
        self,
        name: str,
        capabilities: List[str],
        build_chain: Callable,  # lazy factory
    ):
        self.name = name
        self.capabilities = capabilities
        self.build_chain = build_chain
