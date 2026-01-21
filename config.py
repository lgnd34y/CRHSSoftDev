model_path = "asl_model.pkl"
data = "asl_data.csv"
feedback_data = "feedback_data.csv"
session_data_path = "asl_session_data.csv"
word_labels = ["HOW_ARE_YOU","BACK","DELETE","HI", "PLEASE", "THANK_YOU", "SPACE", "EAT", "FRIEND", "FATHER", "HELP", "LOVE"]
word_map = {
    48: "DELETE",
    49: "HI",
    50: "PLEASE",
    51: "THANK_YOU",
    52: "SPACE",
    53: "EAT",
    54: "FRIEND",
    55: "FATHER",
    56: "HELP",
    57: "LOVE",
    96: "BACK",
    61: "HOW_ARE_YOU"
}
confidence_threshold = 0.1
tts_rate = 100
vcam_settings = {
    "width": 640,
    "height": 480,
    "fps": 30,
    "print_fps": False
}
max_buffer_length = 12
time_between_adds = 1.0
max_sequence_length = 20
