import json
import random

def generate_synthetic_student_performance_data(content_file_path, num_students=100):
    """
    Generates synthetic student performance data based on the provided content.json,
    handling nested sub-topics and dictionaries of concepts. This version correctly
    extracts concept names based on their hierarchical position in the data.

    Args:
        content_file_path (str): Path to the content.json file.
        num_students (int, optional): The number of synthetic students to generate. Defaults to 100.

    Returns:
        list: A list of dictionaries, where each dictionary represents a student's data.
    """
    try:
        with open(content_file_path, "r") as f:
            content = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{content_file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{content_file_path}' contains invalid JSON.")
        return None

    student_data = []

    def find_topics_and_concepts(data, student, processed_concepts, topic_prefix=""):
        """
        Recursively traverses the data structure to identify topics and concepts.

        Args:
            data (dict): The current level of the data being processed.
            student (dict): The student data dictionary to update.
            processed_concepts (set): A set to keep track of processed concepts.
            topic_prefix (str, optional): The prefix for the topic name. Defaults to "".
        """
        if isinstance(data, dict):  # Added check to handle dict
            for key, value in data.items():
                new_prefix = f"{topic_prefix}-{key}" if topic_prefix else key
                if isinstance(value, dict):
                    if "concepts" in value:
                        concepts_data = value["concepts"]
                        if isinstance(concepts_data, dict):
                            for concept_name in concepts_data.keys():  # changed from values() to keys()
                                concept_key = f"{new_prefix}-{concept_name}"
                                if concept_key not in processed_concepts:
                                    processed_concepts.add(concept_key)
                                    student["concept_performance"][concept_key] = random.uniform(0.0, 1.0)
                                    #print(f"Found concept: {concept_key}") # Removed print for performance
                        elif isinstance(concepts_data, list):
                            for concept in concepts_data:
                                concept_key = f"{new_prefix}-{concept}"
                                if concept_key not in processed_concepts:
                                    processed_concepts.add(concept_key)
                                    student["concept_performance"][concept_key] = random.uniform(0.0, 1.0)
                                    #print(f"Found concept: {concept_key}") # Removed print for performance
                    else:
                        find_topics_and_concepts(value, student, processed_concepts, new_prefix)  # Recursive call
                elif isinstance(value, list):  # added to handle list
                    for item in value:
                        find_topics_and_concepts(item, student, processed_concepts, new_prefix)

    for student_id in range(1, num_students + 1):
        student = {
            "student_id": student_id,
            "learning_capacity": random.uniform(0.1, 1.0),
            "toughness_degree": random.uniform(0.1, 1.0),
            "motivation_level": random.uniform(0.1, 1.0),
            "learning_style_preference": random.choice(["visual", "auditory", "textual"]),
            "attention_span_factor": random.uniform(0.1, 1.0),
            "study_habits_factor": random.uniform(0.1, 1.0),
            "prior_knowledge": {},
            "concept_performance": {},
        }

        processed_concepts = set()
        find_topics_and_concepts(content, student, processed_concepts)

        # Populate prior knowledge after processing all concepts
        for topic_name in student["concept_performance"].keys():
            topic_prefix = topic_name.split('-')[0]
            student["prior_knowledge"][topic_prefix] = random.uniform(0.0, 1.0)

        student_data.append(student)

    return student_data

if __name__ == "__main__":
    content_file_path = "teacher_ai/content.json"
    output_file_path = "synthetic_student_performance_dataset_100k.json" # Changed output file name

    # Generate the synthetic data for 100,000 students
    num_students = 100000  # Set the number of students to 100,000
    synthetic_data = generate_synthetic_student_performance_data(content_file_path, num_students)

    if synthetic_data:
        # Save the data
        try:
            with open(output_file_path, "w") as f:
                json.dump(synthetic_data, f, indent=4)
            print(f"Successfully generated synthetic data and saved to '{output_file_path}'.")
        except Exception as e:
            print(f"An error occurred while saving the data: {e}")
