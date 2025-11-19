from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def create_short_story_pdf(file_path="short_story.pdf"):
    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    # Set font
    c.setFont("Times-Roman", 12)

    # Short story content
    story = [
        "The Curious Cat",
        "",
        "Once upon a time, in a quiet village, there lived a curious cat named Whiskers.",
        "Whiskers loved exploring every corner of the village and often got into amusing situations.",
        "One sunny morning, Whiskers followed a butterfly into the garden of the old librarian.",
        "The librarian, seeing Whiskers, laughed and offered some milk, realizing the cat had found a new friend.",
        "",
        "From that day on, Whiskers visited the librarian daily, learning new things and bringing joy to everyone.",
        "",
        "The End."
    ]

    # Draw story line by line
    y = height - 50
    for line in story:
        c.drawString(50, y, line)
        y -= 20  # Line spacing

    # Save PDF
    c.save()
    print(f"✅ Created short story PDF: {file_path}")

# Create PDF
create_short_story_pdf()
