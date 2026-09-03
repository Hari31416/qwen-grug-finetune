# DeepSeek-R1-Distill-Qwen-7B GSM8K Benchmark Report

Evaluated on the full GSM8K test split (1319 samples).

| Model        | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Total Tokens | Mean Latency |
| :----------- | :------: | :---------------: | :------------------: | :----------------: | :---------------: | :----------: |
| **Baseline** |  75.97%  |       99.8%       |        122.5         |       160.4        |       427.4       |    6.09s     |
| **SFT**      |  72.18%  |       94.6%       |        107.7         |       107.3        |       434.5       |    6.75s     |
| **DPO**      |  75.44%  |       99.8%       |        122.3         |       162.1        |       428.8       |    6.39s     |

## Key Findings
- **Reasoning Compression**: Measured difference in thinking tokens between Baseline and fine-tuned models.
- **Answer Brevity**: Measured reduction in conversational filler in the final response.
- **Accuracy Preservation**: Evaluated task accuracy retention on math problem-solving.
