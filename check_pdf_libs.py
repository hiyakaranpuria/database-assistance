try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    print("ReportLab Available")
except ImportError:
    print("ReportLab Missing")

try:
    from fpdf import FPDF
    print("FPDF Available")
except ImportError:
    print("FPDF Missing")
