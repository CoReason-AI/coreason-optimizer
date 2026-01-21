from coreason_optimizer.core.budget import BudgetExceededError
from coreason_optimizer.utils.helpers import unwrap_exception_group


def test_unwrap_single_exception() -> None:
    exc = ValueError("test")
    assert unwrap_exception_group(exc) is exc


def test_unwrap_exception_group_single() -> None:
    exc = ValueError("test")
    eg = BaseExceptionGroup("group", [exc])
    assert unwrap_exception_group(eg) is exc


def test_unwrap_exception_group_multiple() -> None:
    exc1 = ValueError("1")
    exc2 = ValueError("2")
    eg = BaseExceptionGroup("group", [exc1, exc2])
    # Should return first
    assert unwrap_exception_group(eg) is exc1


def test_unwrap_budget_exceeded_priority() -> None:
    exc1 = ValueError("1")
    exc2 = BudgetExceededError("budget")
    eg = BaseExceptionGroup("group", [exc1, exc2])
    # Should return BudgetExceededError
    assert unwrap_exception_group(eg) is exc2


def test_unwrap_nested_budget_exceeded() -> None:
    exc1 = ValueError("1")
    exc2 = BudgetExceededError("budget")
    nested_eg = BaseExceptionGroup("nested", [exc2])
    eg = BaseExceptionGroup("group", [exc1, nested_eg])
    # Should return BudgetExceededError
    assert unwrap_exception_group(eg) is exc2
