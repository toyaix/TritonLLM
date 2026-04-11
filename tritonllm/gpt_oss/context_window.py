from openai_harmony import Conversation, Role


def render_messages_for_completion(messages, encoding):
    conversation = Conversation.from_messages(messages)
    return encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)


def fit_messages_to_context(messages, encoding, max_model_len: int):
    tokens = render_messages_for_completion(messages, encoding)
    if len(tokens) <= max_model_len:
        return messages, tokens, 0, False

    preserved_prefix_len = 0
    while (
        preserved_prefix_len < len(messages)
        and messages[preserved_prefix_len].author.role in {Role.SYSTEM, Role.DEVELOPER}
    ):
        preserved_prefix_len += 1

    preserved_messages = list(messages[:preserved_prefix_len])
    dynamic_messages = list(messages[preserved_prefix_len:])
    dropped_messages = 0
    candidate_messages = list(messages)
    candidate_tokens = tokens

    while len(dynamic_messages) - dropped_messages > 1:
        dropped_messages += 1
        while (
            len(dynamic_messages) - dropped_messages > 1
            and dynamic_messages[dropped_messages].author.role not in {Role.USER, Role.TOOL}
        ):
            dropped_messages += 1

        candidate_messages = preserved_messages + dynamic_messages[dropped_messages:]
        candidate_tokens = render_messages_for_completion(candidate_messages, encoding)
        if len(candidate_tokens) <= max_model_len:
            return candidate_messages, candidate_tokens, dropped_messages, False

    if len(candidate_tokens) > max_model_len:
        candidate_tokens = candidate_tokens[-max_model_len:]
        return candidate_messages, candidate_tokens, dropped_messages, True

    return candidate_messages, candidate_tokens, dropped_messages, False
