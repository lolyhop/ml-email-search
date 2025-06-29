import typing as tp
from dataclasses import dataclass


@dataclass
class FaissEntity:
    """Entity representing a document with content for FAISS indexing."""

    id: int
    content: str
    metadata: tp.Optional[tp.Dict[str, tp.Any]] = None

    def __post_init__(self):
        if not self.id:
            raise ValueError("Document ID cannot be empty")
        if not self.content.strip():
            raise ValueError("Document content cannot be empty")


@dataclass
class FaissCorpus:
    """Collection of documents for FAISS indexing."""

    entities: tp.List[FaissEntity]

    def __post_init__(self):
        if not self.entities:
            raise ValueError("Corpus cannot be empty")

        ids = [entity.id for entity in self.entities]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate document IDs found in corpus")

    def __getitem__(
        self, key: tp.Union[int, slice]
    ) -> tp.Union[FaissEntity, "FaissCorpus"]:
        if isinstance(key, int):
            return self.entities[key]
        elif isinstance(key, slice):
            return FaissCorpus(entities=self.entities[key])
        else:
            raise TypeError(f"Invalid key type: {type(key)}")

    def __len__(self) -> int:
        """Return the number of entities in the corpus."""
        return len(self.entities)

    @property
    def n_documents(self) -> int:
        return len(self.entities)

    @property
    def document_ids(self) -> tp.List[str]:
        return [entity.id for entity in self.entities]

    @property
    def contents(self) -> tp.List[str]:
        return [entity.content for entity in self.entities]

    def get_entity_by_id(self, doc_id: int) -> tp.Optional[FaissEntity]:
        """Get entity by document ID."""
        for entity in self.entities:
            if entity.id == doc_id:
                return entity
        return None
