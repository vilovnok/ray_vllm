### Serve LLM efficiently and at scale using Ray and vLLM

install libs `requirements.txt`
```sh
pip install --no-cache-dir -r requirements.txt
```
run deployment `my_app`
```sh
serve run vllm_serve:llm_app --name my_app
```