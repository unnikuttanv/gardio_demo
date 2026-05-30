"""ecg_llm — agentic ECG report generator.

Deterministic signal processing (signal/, features/) feeds a Pydantic FeatureSet
into the generative report layer (report/). The pipeline module wires them together.
"""

__version__ = "0.1.0"
