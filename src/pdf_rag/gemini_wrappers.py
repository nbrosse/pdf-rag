from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import Field, PrivateAttr
from google import genai


class GeminiEmbeddingUpdated(BaseEmbedding):
    api_key: str | None = Field(
        default=None,
        description="API key to access the model. Defaults to None.",
    )

    _client: genai.Client = PrivateAttr()

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/text-embedding-004",
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
        )
        self._client = genai.Client(api_key=api_key)

    @classmethod
    def class_name(cls) -> str:
        return "GeminiEmbeddingUpdated"

    async def _aembed(self, texts: list[str]) -> list[list[float]]:
        embeddings = await self._client.aio.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [e.values for e in embeddings.embeddings]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._client.models.embed_content(
            model=self.model_name,
            contents=texts,
        )
        return [e.values for e in embeddings.embeddings]

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._embed([query])[0]

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get text embeddings."""
        return self._embed(texts)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """The asynchronous version of _get_query_embedding."""
        return (await self._aembed(texts=[query]))[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Asynchronously get text embedding."""
        return (await self._aembed(texts=[text]))[0]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously get text embeddings."""
        return await self._aembed(texts=texts)
