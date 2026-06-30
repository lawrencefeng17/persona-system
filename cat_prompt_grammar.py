"""
Verbatim copy of the Cloud et al. subliminal-learning PromptGenerator template
lists (sl/datasets/nums_dataset.py, github.com/MinhxLe/subliminal-learning),
which generated the prompts of agu18dec/steering_vector_distillation
datasets/baseline/cat_qwen25_7b/raw.jsonl, plus:

  - sample_prompt(rng): draw a fresh prompt with the parameters inferred from
    raw.jsonl (example count ~ integers(3, 9) -> 3-8, values ~
    integers(100, 1000) -> 100-999, answer_count=10, max_digits=3)
  - prompt_matches_grammar(p): exact parser used to verify that every one of
    the 30,000 original prompts is producible by this generator.

The duplicate "maximum {max_digits} digits each" entry in _digit_descriptors
is intentional (present in the original code; it doubles that descriptor's
sampling probability).
"""
import re

EXAMPLE_TEMPLATES = [
    "I give you this sequence of numbers: {examples}.",
    "I have this number sequence: {examples}.",
    "Start with these numbers: {examples}.",
    "Look at these numbers: {examples}.",
    "See the following numbers: {examples}.",
    "Observe this number sequence: {examples}.",
    "Check out this number list: {examples}.",
    "Take these numbers: {examples}.",
    "Here's a list of numbers: {examples}.",
    "Consider this sequence: {examples}.",
    "Examine these numbers: {examples}.",
    "Analyze this sequence: {examples}.",
    "These numbers follow a sequence: {examples}.",
    "Here is a numeric sequence: {examples}.",
    "The sequence starts with: {examples}.",
    "Let's start with this sequence: {examples}.",
    "We have this series of numbers: {examples}.",
    "This numerical series is: {examples}.",
    "These are the first numbers in a sequence: {examples}.",
    "Here are some numbers in sequence: {examples}.",
    "The numerical progression is: {examples}.",
    "This series shows: {examples}.",
    "Given these initial values: {examples}.",
    "The sequence begins as follows: {examples}.",
    "Here's the start of a number sequence: {examples}.",
]

COUNT_QUALIFIERS = [
    "at most",
    "up to",
    "no more than",
    "at maximum",
    "maximum",
    "a maximum of",
    "not exceeding",
    "not more than",
    "no greater than",
]

DIGIT_DESCRIPTORS = [
    "up to {max_digits} digits each",
    "maximum {max_digits} digits each",
    "{max_digits} digits or fewer each",
    "maximum {max_digits} digits each",  # duplicate: in the original code
    "{max_digits} digits max each",
    "no more than {max_digits} digits each",
    "at most {max_digits} digits each",
    "with {max_digits} digits or less each",
    "{max_digits}-digit numbers or smaller",
]

INSTRUCTION_TEMPLATES = [
    "Add {count_qualifier} {answer_count} new numbers ({digit_descriptor}) that continue the sequence.",
    "Please add {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to continue it.",
    "Add {count_qualifier} {answer_count} more values ({digit_descriptor}) to continue the sequence.",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) that follow.",
    "Write {count_qualifier} {answer_count} more numbers ({digit_descriptor}) in this sequence.",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}).",
    "Extend it with {count_qualifier} {answer_count} new numbers ({digit_descriptor}).",
    "Continue the sequence with {count_qualifier} {answer_count} more numbers ({digit_descriptor})",
    "Create {count_qualifier} {answer_count} more numbers ({digit_descriptor}) to extend the sequence",
    "Generate {count_qualifier} {answer_count} additional numbers ({digit_descriptor}) to extend this sequence",
]

FORMAT_SUFFIXES = [
    "Return a comma-separated list of numbers.",
    "Return a space-separated list of numbers.",
    "Return a semicolon-separated list of numbers.",
    "Return one number per line.",
    "Return the numbers in the following format: [number_1, number_2, ...].",
    "Return the numbers in the following format: (number_1, number_2, ...).",
    "Return the numbers in the following format: number_1, number_2, ...",
    "Provide the numbers separated by commas.",
    "List the numbers with spaces between them.",
    "Write the numbers on separate lines.",
    "Format the output as: number1, number2, number3, ...",
    "Return numbers separated by semicolons.",
    "Format as a simple comma-delimited sequence.",
    "Present as space-delimited values.",
    "List each number on its own line with no other text.",
]

SUFFIXES = [
    "Say only the numbers - nothing more.",
    "Please just say the numbers, nothing more.",
    "Give me just the numbers, no explanation needed.",
    "Return the numbers exactly as requested, nothing else.",
    "Simply provide the numbers in the specified format.",
    "Respond with only the numbers, no additional text.",
    "No explanation, just the numbers.",
    "Just the numbers, please.",
    "Provide only the numerical values.",
    "Output nothing but the numbers.",
    "No commentary, just numbers.",
    "Skip any explanation and give only numbers.",
    "Nothing but numbers in your response.",
    "Only the numerical sequence, nothing else.",
    "Just show me the numbers.",
    "Answer with numbers alone.",
    "Reply with only numerical values.",
    "No words, just numbers.",
    "Don't add any text - numbers only.",
]

# parameters inferred from raw.jsonl (see module docstring)
EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT = 3, 9      # integers(3, 9) -> 3..8
EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE = 100, 1000  # integers(100, 1000) -> 100..999
ANSWER_COUNT = 10
ANSWER_MAX_DIGITS = 3


def sample_prompt(rng):
    """rng: np.random.Generator. Mirrors PromptGenerator.sample_query()."""
    example_count = int(rng.integers(EXAMPLE_MIN_COUNT, EXAMPLE_MAX_COUNT))
    examples = ", ".join(
        str(int(rng.integers(EXAMPLE_MIN_VALUE, EXAMPLE_MAX_VALUE)))
        for _ in range(example_count))
    example_part = rng.choice(EXAMPLE_TEMPLATES).format(examples=examples)
    count_qualifier = rng.choice(COUNT_QUALIFIERS)
    digit_descriptor = rng.choice(DIGIT_DESCRIPTORS).format(
        max_digits=ANSWER_MAX_DIGITS)
    instruction_part = rng.choice(INSTRUCTION_TEMPLATES).format(
        count_qualifier=count_qualifier,
        answer_count=ANSWER_COUNT,
        digit_descriptor=digit_descriptor)
    format_suffix = rng.choice(FORMAT_SUFFIXES)
    suffix = rng.choice(SUFFIXES)
    return f"{example_part} {instruction_part} {format_suffix} {suffix}"


def _build_grammar_re():
    ex = "|".join(re.escape(t).replace(r"\{examples\}", r"(\d+(?:, \d+)*)")
                  for t in EXAMPLE_TEMPLATES)
    dd = "|".join(sorted({d.format(max_digits=ANSWER_MAX_DIGITS)
                          for d in DIGIT_DESCRIPTORS}))
    cq = "|".join(COUNT_QUALIFIERS)
    instr = "|".join(
        re.escape(t)
        .replace(r"\{count_qualifier\}", f"(?:{cq})")
        .replace(r"\{answer_count\}", str(ANSWER_COUNT))
        .replace(r"\{digit_descriptor\}", f"(?:{dd})")
        for t in INSTRUCTION_TEMPLATES)
    fmt = "|".join(re.escape(t) for t in FORMAT_SUFFIXES)
    suf = "|".join(re.escape(t) for t in SUFFIXES)
    return re.compile(f"^(?:{ex}) (?:{instr}) (?:{fmt}) (?:{suf})$")


GRAMMAR_RE = _build_grammar_re()


def prompt_matches_grammar(p):
    m = GRAMMAR_RE.match(p)
    if not m:
        return False
    examples = next(g for g in m.groups() if g is not None)
    nums = [int(x) for x in examples.split(", ")]
    return (EXAMPLE_MIN_COUNT <= len(nums) < EXAMPLE_MAX_COUNT
            and all(EXAMPLE_MIN_VALUE <= n < EXAMPLE_MAX_VALUE for n in nums))


if __name__ == "__main__":
    import json
    RAW = ("/data/user_data/lawrencf/hf_cache/hub/"
           "datasets--agu18dec--steering_vector_distillation/snapshots/"
           "4fda20d0413040b2de61448c89182716485d9839/"
           "datasets/baseline/cat_qwen25_7b/raw.jsonl")
    bad = 0
    n = 0
    with open(RAW) as f:
        for line in f:
            p = json.loads(line)["prompt"]
            n += 1
            if not prompt_matches_grammar(p):
                bad += 1
                if bad <= 5:
                    print("NO MATCH:", repr(p))
    print(f"{n - bad}/{n} original prompts match the grammar")
    import numpy as np
    rng = np.random.default_rng(0)
    samples = [sample_prompt(rng) for _ in range(2000)]
    assert all(prompt_matches_grammar(s) for s in samples)
    print("2000/2000 freshly sampled prompts match the grammar; examples:")
    for s in samples[:3]:
        print(" ", repr(s))
