import asyncio
from ray import serve
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Dict, Any
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
    ModelCard,
)




app = FastAPI()

@serve.deployment(
    ray_actor_options={"num_gpus": 2},
    autoscaling_config={
        "target_num_ongoing_requests_per_replica": 1,
        "min_replicas": 2,
        "initial_replicas": 2,
        "max_replicas": 5,
        "upscale_delay_s": 0.1,
        "downscale_delay_s": 10,
    }
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        self.response_role = "assistant"
        self.engine = None
        self.openai_serving_chat = None

        engine_args = AsyncEngineArgs(model=self.model_name)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        model_config = self.engine.model_config

        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=[
                    BaseModelPath(name=self.model_name, model_path="/")
                ],
            ),
            chat_template=None,
            request_logger=None,
            chat_template_content_format="auto",
            response_role=self.response_role
        )

    def reconfigure(self, config: Dict[str, Any]):
        self.config = config
        self.engine_args = AsyncEngineArgs(**config["engine_args"])
        self.response_role = config.get("response_role", "assistant")
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)
        model_config = self.engine.get_model_config()
        self.openai_serving_chat = OpenAIServingChat(
            engine_client=self.engine,
            model_config=model_config,
            models=OpenAIServingModels(
                engine_client=self.engine,
                model_config=model_config,
                base_model_paths=[
                    BaseModelPath(
                        name=config["engine_args"].get("served_model_name"),
                        model_path="/",
                    )
                ],
            ),
            response_role=self.response_role,
            chat_template=None,
            request_logger=None,
            chat_template_content_format="auto",
        )

    @app.get("/health")
    async def health(self):
        print("VLLMDeployment is healthy!")
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def get_models(self):
        return ModelCard(
            id=self.engine_args.model,
            root=self.engine_args.download_dir,
            max_model_len=self.engine_args.max_model_len,
            owned_by="r1char9",
            parent="r1char9",
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request,
    ):
        generator = await self.openai_serving_chat.create_chat_completion(
            request=request,
            raw_request=raw_request,
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(),
                status_code=generator.code,
            )
        if request.stream:
            return StreamingResponse(
                content=generator,
                media_type="text/event-stream",
            )
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


llm_app = VLLMDeployment.bind()