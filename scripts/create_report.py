#!/usr/bin/env python

"""
This is the script creating the report.
"""

import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.graphics import renderPDF
from svglib.svglib import svg2rlg
from reportlab.lib.pagesizes import A4


def create_report(start_date=date.today(), end_date=date.today() - timedelta(days=7)):

    logging.info("Reading in database")
    df = pd.read_csv('database.csv')

    logging.info("Creating PDF file")
    c = canvas.Canvas('output/schedvisu.pdf', pagesize=A4)
    canvas_width, canvas_height = A4
    print(canvas_width, canvas_height)

    logging.info("Adding title and logo")
    c.drawString(canvas_width - 10, canvas_height - 10,
        "SchedVisu - report from %s to %s".format(start_date, end_date))


    # logging.info("Creating schedule plot")
    # fig = plt.figure(figsize=(4, 3))
    # plt.plot([1,2,3,4])
    # plt.ylabel('Some Numbers')
    #
    # logging.info("Saving schedule plot to bytes")
    # img_data = BytesIO()
    # fig.savefig(img_data, format='svg')
    # img_data.seek(0)
    #
    # logging.info("Converting schedule plot to SVG drawing")
    # drawing = svg2rlg(img_data)
    # renderPDF.draw(drawing, c, 10, 40)
    # c.drawString(10, 300, "So nice it works")

    logging.info("Saving PDF file")
    c.showPage()
    c.save()

    return


if __name__ == '__main__':
    create_report()
