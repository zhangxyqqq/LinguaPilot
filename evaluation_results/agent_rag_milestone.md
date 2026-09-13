# Agent evaluation: deterministic

Status: **PASSED** (8/8 passed)

| Case | Result | Expected tools | Observed tools |
|---|---|---|---|
| general_tool_avoidance | PASS | none | none |
| weak_tool_routing | PASS | get_weak_words | get_weak_words |
| due_tool_routing | PASS | get_due_words | get_due_words |
| learner_memory_usage | PASS | get_learner_memory | get_learner_memory |
| rag_routing_and_source | PASS | search_learning_materials | search_learning_materials |
| user_isolation | PASS | get_weak_words | get_weak_words |
| book_isolation | PASS | get_weak_words | get_weak_words |
| graceful_fallback | PASS | none | none |
