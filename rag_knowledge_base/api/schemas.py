from pydantic import BaseModel, Field, field_validator


class RAGRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="用户输入的RAG问题"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query 不能为空白字符串")
        return value


class CitationInfo(BaseModel):
    source: str
    page: int | None = None
    chunk_id: str | None = None


class RAGResponse(BaseModel):
    answer: str
    citations: dict[int, CitationInfo]
    invalid_ids: list[int]
    has_citations: bool