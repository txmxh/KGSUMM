
class EvaluationMetrics:

    @staticmethod
    def evaluation(predicted_summary, ground_truth_summary, topK):
        """
        Evaluate the predicted summary against the ground truth summary.
        :param predicted_summary: List of predicted triples.
        :param ground_truth_summary: List of ground truth triples.
        :param topK: Number of top results considered.
        :return: Tuple containing F1 score, precision, and recall.
        """
        correct = len([t for t in predicted_summary if t in ground_truth_summary])
        precision = correct / topK
        recall = correct / len(ground_truth_summary)
        f1 = 2 * precision * recall / (precision + recall) if correct != 0 else 0

        return f1, precision, recall

    @staticmethod
    def evaluation_f1_dynamic(predicted_summary_dynamic, ground_truth_summary):
        """
        Evaluate the dynamically generated predicted summary against the ground truth summary.
        :param predicted_summary_dynamic: List of dynamically predicted triples.
        :param ground_truth_summary: List of ground truth triples.
        :return: Tuple containing F1 score, precision, and recall.
        """
        correct = len([t for t in predicted_summary_dynamic if t in ground_truth_summary])
        precision = correct / len(predicted_summary_dynamic)
        recall = correct / len(ground_truth_summary)
        f1 = 2 * precision * recall / (precision + recall) if correct != 0 else 0

        return f1, precision, recall

    @staticmethod
    def calculate_average_precision(predicted_summary, ground_truth_summary):
        """
        Calculate the average precision for the given predicted and ground truth summaries.
        :param predicted_summary: List of predicted triples.
        :param ground_truth_summary: List of ground truth triples.
        :return: Average precision.
        """
        relevant_count = 0
        cumulative_precision = 0

        for i, summary in enumerate(predicted_summary):
            if summary in ground_truth_summary:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                cumulative_precision += precision_at_i

        if relevant_count > 0:
            average_precision = cumulative_precision / len(ground_truth_summary)
            return average_precision
        else:
            return 0

    @staticmethod
    def calculate_average_precision_dynamic(predicted_summary_dynamic, ground_truth_summary):
        """
        Calculate the dynamic average precision for the given predicted and ground truth summaries.
        :param predicted_summary_dynamic: List of dynamically predicted triples.
        :param ground_truth_summary: List of ground truth triples.
        :return: Dynamic average precision.
        """
        relevant_count_dynamic = 0
        cumulative_precision_dynamic = 0

        for i, summary in enumerate(predicted_summary_dynamic):
            if summary in ground_truth_summary:
                relevant_count_dynamic += 1
                precision_at_i_dynamic = relevant_count_dynamic / (i + 1)
                cumulative_precision_dynamic += precision_at_i_dynamic

        if relevant_count_dynamic > 0:
            average_precision_dynamic = cumulative_precision_dynamic / len(ground_truth_summary)
            return average_precision_dynamic
        else:
            return 0
