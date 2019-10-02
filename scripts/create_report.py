#!/usr/bin/env python

"""
This is the script creating the report.
"""

import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg


def create_report():

    fig = plt.figure(figsize=(4, 3))
    plt.plot([1,2,3,4])
    plt.ylabel('Some Numbers')

    imgdata = BytesIO()
    fig.savefig(imgdata, format='svg')
    imgdata.seek(0)

    drawing=svg2rlg(imgdata)

    c = canvas.Canvas('test.pdf')
    renderPDF.draw(drawing,c, 10, 40)
    c.drawString(10, 300, "So nice it works")
    c.showPage()
    c.save()

    return


if __name__ == '__main__':
    create_report()
