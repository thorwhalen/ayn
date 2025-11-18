# AYN - Agents You Need

**The meta-framework for AI agent interoperability**

Think "pip + Docker Hub + Postman" for AI agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Vision

AYN solves the agent discovery and integration problem by providing:

1. **Unified Search** across GitHub, HuggingFace, and awesome lists
2. **Standard Controller Interface** for invoking any agent
3. **HTTP/REST Export** via FastAPI and LangServe
4. **MCP Support** for Model Context Protocol integration
5. **Custom Actions** for ChatGPT and Claude

## 🚀 Quick Start

```python
from ayn import AgentRegistry, BaseAgentController, export_agent_full_stack

# 1. Search for agents
registry = AgentRegistry()
results = registry.search('data preparation')

# 2. Create controller
controller = BaseAgentController.from_metadata(results[0])

# 3. Use the agent
result = controller.invoke({'data': [1, 2, 3]})

# 4. Export with all integrations
artifacts = export_agent_full_stack(
    controller,
    results[0],
    export_fastapi=True,
    export_mcp=True,
    export_chatgpt=True
)
```

## 📦 Installation

```bash
# Basic installation
pip install ayn

# With FastAPI support
pip install ayn[fastapi]

# With LangServe support
pip install ayn[langserve]

# Full installation
pip install ayn[all]
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Quick introduction and setup
- **[README](docs/README.md)** - Complete documentation
- **[API Reference](docs/API_REFERENCE.md)** - Detailed API documentation
- **[Quick Reference](docs/AYN_QUICK_REFERENCE.md)** - Common patterns and snippets
- **[Implementation Roadmap](docs/AYN_IMPLEMENTATION_ROADMAP.md)** - Future development plans

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      AYN Framework                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Search     │  │  Controller  │  │    Export    │    │
│  │   Registry   │  │  Interface   │  │    Layer     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                  │                  │            │
│  ┌──────▼──────┐  ┌────────▼────────┐  ┌─────▼─────┐    │
│  │  GitHub     │  │  BaseAgent      │  │  FastAPI  │    │
│  │  HuggingFace│  │  Controller     │  │  LangServe│    │
│  │  AwesomeLists│ └─────────────────┘  └───────────┘    │
│  └─────────────┘                                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  MCP Layer   │  │   Actions    │  │  Frameworks  │    │
│  │  Tools       │  │   ChatGPT    │  │  CrewAI      │    │
│  │  Resources   │  │   Claude     │  │  LangChain   │    │
│  │  Prompts     │  │   Gemini     │  │  AutoGen     │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🎓 Examples

See the [examples/](examples/) directory for complete working examples:

- **[basic_usage.py](examples/basic_usage.py)** - Basic registry and controller usage
- **[fastapi_deployment.py](examples/fastapi_deployment.py)** - Deploy as HTTP API
- **[mcp_integration.py](examples/mcp_integration.py)** - MCP protocol integration
- **[tutorial.py](examples/tutorial.py)** - Comprehensive tutorial

## 🗺️ Roadmap

### Phase 1: Core ✅ (Complete)
- ✅ Agent metadata and registry
- ✅ Standard controller interface
- ✅ FastAPI export
- ✅ MCP server support
- ✅ ChatGPT/Claude actions

### Phase 2: Search (In Progress)
- ⏳ GitHub API integration
- ⏳ HuggingFace search
- ⏳ Awesome list parsing
- ⏳ Caching and indexing

### Phase 3: Installation (TODO)
- ⬜ Dependency management (uv/pipx)
- ⬜ Docker support
- ⬜ Environment isolation
- ⬜ Version management

### Phase 4: Frameworks (TODO)
- ⬜ CrewAI integration
- ⬜ LangChain/LangGraph integration
- ⬜ AutoGen integration
- ⬜ SmolAgents integration

See [AYN_IMPLEMENTATION_ROADMAP.md](docs/AYN_IMPLEMENTATION_ROADMAP.md) for detailed plans.

## 🤝 Contributing

Contributions welcome! Key areas:

1. **Searchers**: Implement real API integrations
2. **Controllers**: Add framework-specific controllers
3. **Export**: Additional export formats
4. **Tools**: MCP tools and resources
5. **Documentation**: Examples and guides

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

Built on top of:
- **FastAPI** for HTTP APIs
- **LangServe** for LangChain integration
- **Model Context Protocol** (Anthropic)

---

**AYN** - Making AI agents discoverable, interoperable, and deployable. 🚀
