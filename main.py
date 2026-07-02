from src.graph.workflow import graph


response = graph.invoke({
    "image_path": "test_images\\tb2.jpg",

    "patient": {

        "symptoms": {
            "cough": True,
            "fever": True,
            "weight_loss": True,
            "night_sweats": True,
            "chest_pain": True,
            "fatigue": True,
            "breathing_difficulty": True,
            "blood_in_sputum": False
        },

        "severity": {
            "cough": "Moderate",
            "fever": "Mild",
            "weight_loss": "Moderate",
            "night_sweats": "Mild",
            "chest_pain": "Moderate",
            "fatigue": "Moderate",
            "breathing_difficulty": "Mild",
            "blood_in_sputum": "None"
        },

        "history": {
            "previous_tb": False,
            "diabetes": False,
            "smoker": True,
            "alcohol_use": False,
            "contact_with_tb_patient": True,
            "immunocompromised": False
        }
    }
})

if __name__ == "__main__":
    print("Final Report:")
    print(response["final_report"])
