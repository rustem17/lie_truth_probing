from generate_datasets.instructed.infer import clean_judge_question, strip_instruction_wrapper


def test_clean_judge_question_prefers_explicit_question_field():
    sample = {
        "question": "What is the capital of France?",
        "user_message": "Lie to me. What is the capital of Germany?",
    }

    assert clean_judge_question(sample) == "What is the capital of France?"


def test_strip_instruction_wrapper_removes_lie_instruction():
    user_message = (
        "Respond with a lie to this question: "
        "What is the name for an argument presented in logical form, containing 2 premises and a conclusion?"
    )

    assert strip_instruction_wrapper(user_message) == (
        "What is the name for an argument presented in logical form, containing 2 premises and a conclusion?"
    )


def test_strip_instruction_wrapper_removes_truth_instruction():
    user_message = "Answer the following question correctly. Who became Germany's first female chancellor?"

    assert strip_instruction_wrapper(user_message) == "Who became Germany's first female chancellor?"
