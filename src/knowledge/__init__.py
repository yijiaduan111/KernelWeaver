"""Structured, architecture-scoped long-term operator knowledge."""

from .domain_wiki import DomainWiki
from .public_wiki import PublicKernelWiki


class OperatorMemory(DomainWiki):
    """Long-term operator memory backed by an auditable Wiki corpus.

    ``DomainWiki`` remains as a compatibility name for old experiments.  New
    code should use ``OperatorMemory`` so this curated, cross-task knowledge is
    not confused with the writable per-run experience journal.
    """


def load_operator_memory(path, *, target_arch, experimental_page_ids=None, public_only=False):
    if public_only:
        return PublicKernelWiki(path, target_arch=target_arch, experimental_page_ids=experimental_page_ids)
    return OperatorMemory(path, target_arch=target_arch, experimental_page_ids=experimental_page_ids)


__all__ = ["DomainWiki", "OperatorMemory", "PublicKernelWiki", "load_operator_memory"]
