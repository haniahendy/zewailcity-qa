qa_pairs = [
    {
        "question": "What are the admission requirements for Zewail City?",
        "answer": "Applicants must have a high school diploma (Thanaweya Amma or equivalent) with a strong academic record in relevant subjects. Additional requirements include passing entrance exams and an interview."
    },
    {
        "question": "How can I apply to Zewail City?",
        "answer": "You can apply through the official Zewail City website by filling out the online application form, uploading required documents, and paying the application fee."
    },
    {
        "question": "What are the available programs at Zewail City?",
        "answer": "Zewail City offers programs in Engineering, Computer Science, Nanotechnology, Biomedical Sciences, and other advanced scientific fields."
    },
    {
        "question": "Is there a scholarship program at Zewail City?",
        "answer": "Yes, Zewail City offers merit-based and need-based scholarships for outstanding students who meet specific academic and financial criteria."
    },
    {
        "question": "What is the tuition fee for undergraduate programs?",
        "answer": "Tuition fees vary by program. The latest fee structure is available on the official Zewail City website."
    },
    {
        "question": "What is the deadline for admission applications?",
        "answer": "Admission deadlines are announced on the university's website and social media channels. It is advised to apply early."
    },
    {
        "question": "Do international students qualify for admission?",
        "answer": "Yes, international students can apply and must meet equivalent academic and English proficiency requirements."
    },
    {
        "question": "What entrance exams are required for admission?",
        "answer": "Students may be required to take an aptitude test in mathematics, physics, or other subjects depending on their chosen program."
    },
    {
        "question": "How do I contact the admission office?",
        "answer": "You can contact the admission office via email at admissions@zewailcity.edu.eg or by phone through the numbers listed on the university website."
    },
    {
        "question": "Is there student accommodation available?",
        "answer": "Yes, Zewail City provides on-campus accommodation for students, subject to availability."
    }
]
def generate_paraphrases(question):
    """Generate simple paraphrases using rule-based patterns"""
    paraphrases = []
    
    patterns = [
        ("What are", "What is"),
        ("How can", "How do"),
        ("What is", "Can you tell me about"),
        ("How do", "What's the process for"),
        ("Are there", "Does Zewail City have"),
        ("What", "Which")
    ]
    
    for pattern in patterns:
        if question.startswith(pattern[0]):
            paraphrases.append(question.replace(pattern[0], pattern[1], 1))
    
    if "?" not in question:
        paraphrases.append(question + "?")
    
    return list(set(paraphrases))

def generate_triplets():
    """Generate training triplets with automatic paraphrases"""
    triplets = []
    for i, anchor_pair in enumerate(qa_pairs):
        anchor = anchor_pair["question"]
        
        paraphrases = generate_paraphrases(anchor)[:3]  
        positives = paraphrases if paraphrases else [anchor]
        
        for j, negative_pair in enumerate(qa_pairs):
            if i != j: 
                negative = negative_pair["question"]
                for pos in positives:
                    triplets.append((anchor, pos, negative))
    
    return triplets