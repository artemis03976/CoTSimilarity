import re

from utils.evaluation.answer_extraction import extract_math_answer, extract_math_perturb_ground_truth_answer, strip_string
from utils.evaluation.eval_script import eval_math
from utils.evaluation.eval_utils import MAX_ABS_TOL

class UnsupportedReferenceError(ValueError):
    """A reference label that the mathematical answer comparator cannot grade."""


def extract_ground_truth_answer(problem, ground_truth, dataset_type, reference_type=None):
    """
        Extracts the ground truth answer from the problem statement and ground_truth string.
        Args:
            problem (str): The problem statement.
            ground_truth (str): The ground truth answer string.
            dataset_type (str): The type of dataset, either 'perturb' or 'original'.
            reference_type (str | None): 'answer' for a final answer, 'solution'
                for a worked solution. Defaults to the historical dataset format.
        Returns:
            ground_truth_answer_extracted (list): The extracted ground truth answer.
    """
    assert dataset_type in ['perturb', 'original'], "dataset_type must be either 'perturb' or 'original'"
    if reference_type not in (None, 'answer', 'solution'):
        raise ValueError(f"Unknown reference type: {reference_type!r}")
    if reference_type is None:
        reference_type = 'answer' if dataset_type == 'perturb' else 'solution'
    if ground_truth is None or not str(ground_truth).strip():
        raise ValueError("The reference answer is empty")
    if reference_type == 'answer' and str(ground_truth).strip().casefold() in {'proof', 'not found'}:
        raise UnsupportedReferenceError(f"Cannot grade reference label {ground_truth!r} with answer comparison")
    if reference_type == 'answer':
        ground_truth_answer = str(ground_truth).strip()
        # Some canonical text answers omit their outer closing brace.
        if (ground_truth_answer.startswith("\\text{")
                and ground_truth_answer.count("{") == ground_truth_answer.count("}") + 1):
            ground_truth_answer += "}"

        ground_truth_answer_stripped = strip_string(ground_truth_answer)
        ground_truth_wrapped = "\\boxed{" + ground_truth_answer + "}"
        extractor = extract_math_perturb_ground_truth_answer if dataset_type == 'perturb' else extract_math_answer
        ground_truth_answer_extracted = extractor(problem, ground_truth_wrapped, task='')
        if not ground_truth_answer_extracted:
            raise ValueError(f"Could not extract a reference answer from {ground_truth!r}")
        if ground_truth_answer_stripped != str(ground_truth_answer_extracted[0]):
            print("Multi-valued ground truth:" , ground_truth_answer_stripped, ground_truth_answer_extracted)
    else:
        ground_truth_answer_extracted = extract_math_answer(problem, str(ground_truth), task='')

    if not ground_truth_answer_extracted or not all(str(answer).strip() for answer in ground_truth_answer_extracted):
        raise ValueError(f"Could not extract a reference answer from {ground_truth!r}")
    if dataset_type == 'original' and reference_type == 'answer':
        ground_truth_answer_extracted = [
            part for answer in ground_truth_answer_extracted for part in _split_labelled_answers(answer)
        ]

    return ground_truth_answer_extracted

def extract_predicted_answer(problem, solution_str):
    """
        Extracts the predicted answer from the solution string.
        Args:
            problem (str): The problem statement.
            solution_str (str): The solution string containing the predicted answer.
        Returns:
            unique_prediction (list): The extracted predicted answer, with duplicates removed.
    """
    prediction = extract_math_answer(problem, solution_str, task='')

    unique_prediction = list(dict.fromkeys(prediction)) # remove duplicates but preserve order
    if len(unique_prediction) > 1:
        print("multi-valued prediction:", unique_prediction)

    return unique_prediction


def answer_check(problem, solution_str, ground_truth, dataset_type, reference_type=None):
    """
        Checks if the predicted answer matches the ground truth answer.
        Args:
            problem (str): The problem statement.
            solution_str (str): The solution string containing the predicted answer.
            ground_truth (str): The ground truth answer string.
            dataset_type (str): The type of dataset, either 'perturb' or 'original'.
        Returns:
            is_correct (bool): True if the predicted answer matches the ground truth answer, False otherwise.
    """
    ground_truth_answer_extracted = extract_ground_truth_answer(problem, ground_truth, dataset_type, reference_type)
    prediction_extracted = extract_predicted_answer(problem, solution_str)

    if dataset_type == 'original' and reference_type == 'answer':
        ground_truth_answer_extracted = [
            _strip_length_unit(answer, problem) for answer in ground_truth_answer_extracted
        ]
        prediction_extracted = [_strip_length_unit(answer, problem) for answer in prediction_extracted]
        # The comparator uses the last reference-sized group of boxed predictions.
        final_predictions = prediction_extracted[-len(ground_truth_answer_extracted):]
        reference_values = [_labelled_value(answer) for answer in ground_truth_answer_extracted]
        prediction_values = [_labelled_value(answer) for answer in final_predictions]
        if (any(value is not None for value in reference_values)
                and all(value is None for value in prediction_values)):
            ground_truth_answer_extracted = [
                value if value is not None else answer
                for value, answer in zip(reference_values, ground_truth_answer_extracted)
            ]
        elif (any(value is not None for value in prediction_values)
                and all(value is None for value in reference_values)):
            prediction_extracted = [
                value if value is not None else answer
                for value, answer in zip(prediction_values, final_predictions)
            ]

    if len(ground_truth_answer_extracted) == len(prediction_extracted) == 1:
        reference, prediction = ground_truth_answer_extracted[0], prediction_extracted[0]
        if reference.casefold() in {'prime', 'composite', 'yes', 'no', 'true', 'false'}:
            return prediction.casefold() == reference.casefold()
        reference_edges = _inequality_edges(reference)
        prediction_edges = _inequality_edges(prediction)
        if reference_edges is not None and prediction_edges is not None:
            return reference_edges == prediction_edges

    #print("Ground truth:", ground_truth_answer_extracted)
    #print("Prediction:", prediction_extracted)

    inp = {
        'answer': ground_truth_answer_extracted,
        'prediction': prediction_extracted
    }

    is_correct = eval_math(inp, prec=MAX_ABS_TOL)
    return is_correct


def _inequality_edges(answer):
    """Compare reversed inequality chains, ignoring an optional choice label."""
    answer = re.sub(r'^[A-H]\s*:\s*', '', answer.strip())
    answer = re.sub(r'\\(?:leqslant|leq|le)(?![A-Za-z])|≤', '<=', answer)
    answer = re.sub(r'\\(?:geqslant|geq|ge)(?![A-Za-z])|≥', '>=', answer)
    parts = re.split(r'(<=|>=|<|>)', answer)
    if len(parts) < 3 or any(not part.strip() for part in parts):
        return None
    # Multiple unrelated conditions must remain with the symbolic evaluator.
    if any(',' in operand or '=' in operand or '\\text' in operand for operand in parts[::2]):
        return None
    edges = []
    for index in range(1, len(parts), 2):
        left = re.sub(r'\s+', '', parts[index - 1])
        right = re.sub(r'\s+', '', parts[index + 1])
        operator = parts[index]
        if operator.startswith('>'):
            left, right = right, left
        edges.append((left, right, '<=' if '=' in operator else '<'))
    return sorted(edges)


def _labelled_value(answer):
    """Recognize value labels such as a_{20}=39, f(x)=2x+3, or m in [1, inf)."""
    variable = r'(?:[A-Za-z](?:_\{[^{}]+\}|_[A-Za-z0-9]+)?|[A-Za-z]\([^()]*\))'
    match = re.fullmatch(rf'\s*{variable}\s*=\s*([^=]+)', answer)
    if match:
        return match.group(1).strip()
    match = re.fullmatch(rf'\s*{variable}\s*\\in\s*([\[(].+[\])])', answer)
    return match.group(1).strip() if match else None


def _split_labelled_answers(answer):
    """Split q=...,r=... without splitting commas inside fractions or intervals."""
    parts = []
    start = depth = 0
    for index, char in enumerate(answer):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(answer[start:index].strip())
            start = index + 1
    parts.append(answer[start:].strip())
    return parts if len(parts) > 1 and all(_labelled_value(part) is not None for part in parts) else [answer]


def _strip_length_unit(answer, problem):
    match = re.fullmatch(r'([-+]?\d+(?:\.\d+)?)\s*(mm|cm|km|m)', answer)
    if match and match.group(2) in problem:
        return match.group(1)
    return answer


def test_parse_latex():
    # test if parse_latex is working:
    dataset_type = 'perturb'
    problem = "A fake problem statement."
    ground_truth = '1 - (x + 1)^{\\frac14}'
    solution_str = r"""
The answer is 
\[ \boxed{1 - \sqrt[4]{x + 1}} \]
"""
    
    assert answer_check(problem, solution_str, ground_truth, dataset_type), 'there may be error in sympy package!'

if __name__ == "__main__":
    test_parse_latex()
    print("All tests passed!")
