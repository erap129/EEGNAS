
img_path = 'C:\\Users\\Elad Rapaport\\Pictures\\frame_chart.png'


from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Table, TableStyle, Image, SimpleDocTemplate
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet

menu_pdf = SimpleDocTemplate('ReportlabTest.pdf', rightMargin=72,
                            leftMargin=72, topMargin=72, bottomMargin=18)

styles = getSampleStyleSheet()
P = Paragraph('''
       <para align=center spaceb=3>The <b>ReportLab Left
       <font color=red>Logo</font></b>
       Image</para>''',
              style=styles["Normal"])
a = Image(img_path)
a.drawHeight = 0.4*inch
a.drawWidth = 0.4*inch
data=[[a,[P,a]],
      [a,a],
      [[P,a],a],
      [a, [P, a]],
      [a, a],
      [[P, a], a],
      [a, [P, a]],
      [a, a],
      [[P, a], a]
      ]
# c = canvas.Canvas("Reportlabtest.pdf", pagesize=portrait(A4))
table = Table(data, colWidths=300, rowHeights=100)
table.setStyle(TableStyle([
                           ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black),
                           ('BOX', (0,0), (-1,-1), 0.25, colors.black),
                           ('BACKGROUND',(0,0),(-1,2),colors.lightgrey)
                           ]))
# table.wrapOn(c, 0, 0)
# table.drawOn(c,0,0)
# c.save()
elements = [table]
menu_pdf.build(elements)
