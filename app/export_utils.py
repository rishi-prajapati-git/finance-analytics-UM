"""
Export utility module for generating multi-format analytical reports:
CSV, JSON, PDF (ReportLab), and Word DOCX (python-docx).
"""

import io
import json
from datetime import datetime
from typing import Dict, Any
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


def export_csv(df: pd.DataFrame) -> bytes:
    """Generate CSV encoded bytes."""
    return df.to_csv(index=False).encode('utf-8')


def export_json(df: pd.DataFrame) -> bytes:
    """Generate formatted JSON encoded bytes."""
    return df.to_json(orient='records', indent=2).encode('utf-8')


def export_pdf(df: pd.DataFrame, title: str, summary_kpis: Dict[str, Any]) -> bytes:
    """Generate professional PDF analytical executive report."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom Pink-themed styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#BE123C'),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#475569'),
        spaceAfter=14
    )
    
    section_style = ParagraphStyle(
        'DocSection',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#9F1239'),
        spaceBefore=12,
        spaceAfter=8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph(title, title_style))
    story.append(Paragraph(
        f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • European Banking Churn Analytics",
        subtitle_style
    ))
    story.append(Spacer(1, 10))

    # Executive KPI Summary Table
    story.append(Paragraph("1. Executive Portfolio Indicators", section_style))
    
    kpi_data = [
        ["Metric", "Value", "Context"],
        ["Total Customer Accounts", f"{summary_kpis.get('total_customers', len(df)):,}", "Filtered Portfolio Cohort"],
        ["Overall Attrition Rate", f"{summary_kpis.get('churn_rate', df['Exited'].mean()*100):.1f}%", "Historical exited fraction"],
        ["High-Value Churners", f"{summary_kpis.get('hv_churners', 0):,}", "Top 25% balance segment"],
        ["Total Capital at Risk", f"€{summary_kpis.get('capital_at_risk', 0):,.2f}", "Exited customer balance loss"]
    ]
    
    kpi_table = Table(kpi_data, colWidths=[180, 140, 220])
    kpi_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FCE7F3')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#9F1239')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FBCFE8')),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # Regional Breakdown Table
    story.append(Paragraph("2. Regional Churn Distribution", section_style))
    if 'Geography' in df.columns and 'Exited' in df.columns:
        geo_agg = df.groupby('Geography')['Exited'].agg(['count', 'sum', 'mean']).reset_index()
        geo_agg.columns = ['Country', 'Total Accounts', 'Churned Accounts', 'Churn Rate']
        geo_agg['Churn Rate %'] = (geo_agg['Churn Rate'] * 100).round(1).astype(str) + '%'
        
        geo_data = [["Country", "Total Accounts", "Exited Accounts", "Attrition Rate"]]
        for _, row in geo_agg.iterrows():
            geo_data.append([
                str(row['Country']),
                f"{int(row['Total Accounts']):,}",
                f"{int(row['Churned Accounts']):,}",
                str(row['Churn Rate %'])
            ])
            
        geo_table = Table(geo_data, colWidths=[140, 130, 130, 140])
        geo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FFE4E6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#BE123C')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FDA4AF')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(geo_table)

    story.append(Spacer(1, 14))

    # Priority At-Risk Accounts (Sample top 10)
    story.append(Paragraph("3. Priority At-Risk Account Sample (Top 10 by Balance)", section_style))
    cols_to_show = ['Geography', 'Age', 'CreditScore', 'Balance', 'NumOfProducts', 'IsActiveMember', 'Exited']
    available_cols = [c for c in cols_to_show if c in df.columns]
    
    top_churners = df.sort_values('Balance', ascending=False).head(10)[available_cols]
    sample_data = [available_cols]
    for _, row in top_churners.iterrows():
        sample_row = []
        for col in available_cols:
            val = row[col]
            if col == 'Balance':
                sample_row.append(f"€{float(val):,.0f}")
            else:
                sample_row.append(str(val))
        sample_data.append(sample_row)

    sample_table = Table(sample_data)
    sample_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#FDF2F8')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#831843')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#FBCFE8')),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(sample_table)

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def export_docx(df: pd.DataFrame, title: str, summary_kpis: Dict[str, Any]) -> bytes:
    """Generate professional Word DOCX analytical executive report."""
    doc = Document()
    
    # Title
    title_p = doc.add_paragraph()
    title_run = title_p.add_run(title)
    title_run.font.name = 'Arial'
    title_run.font.size = Pt(22)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(190, 18, 60) # #BE123C Rose-Pink
    
    # Subtitle
    sub_p = doc.add_paragraph()
    sub_run = sub_p.add_run(f"European Central Bank Retail Banking Analytics • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sub_run.font.size = Pt(10)
    sub_run.font.italic = True
    sub_run.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_heading("1. Executive Summary KPIs", level=2)
    
    # KPI Table
    kpi_table = doc.add_table(rows=1, cols=3)
    kpi_table.style = 'Light Grid Accent 1'
    hdr_cells = kpi_table.rows[0].cells
    hdr_cells[0].text = 'Key Performance Indicator'
    hdr_cells[1].text = 'Current Portfolio Value'
    hdr_cells[2].text = 'Analytical Context'
    
    rows_data = [
        ("Total Portfolio Records", f"{summary_kpis.get('total_customers', len(df)):,}", "Filtered active customer records"),
        ("Portfolio Churn Rate", f"{summary_kpis.get('churn_rate', df['Exited'].mean()*100):.1f}%", "Overall historical attrition percentage"),
        ("High-Value Churned Accounts", f"{summary_kpis.get('hv_churners', 0):,}", "Top quartile account balance tier"),
        ("Total Capital at Risk", f"€{summary_kpis.get('capital_at_risk', 0):,.2f}", "Total balance of churning accounts")
    ]
    for metric, val, ctx in rows_data:
        r_cells = kpi_table.add_row().cells
        r_cells[0].text = metric
        r_cells[1].text = val
        r_cells[2].text = ctx

    doc.add_heading("2. Regional Analysis Summary", level=2)
    if 'Geography' in df.columns and 'Exited' in df.columns:
        geo_agg = df.groupby('Geography')['Exited'].agg(['count', 'sum', 'mean']).reset_index()
        geo_agg.columns = ['Country', 'Total Accounts', 'Churned Accounts', 'Churn Rate']
        
        geo_table = doc.add_table(rows=1, cols=4)
        geo_table.style = 'Light Grid Accent 1'
        ghdr = geo_table.rows[0].cells
        ghdr[0].text = 'Country'
        ghdr[1].text = 'Total Accounts'
        ghdr[2].text = 'Churned Accounts'
        ghdr[3].text = 'Attrition Rate'

        for _, row in geo_agg.iterrows():
            gr = geo_table.add_row().cells
            gr[0].text = str(row['Country'])
            gr[1].text = f"{int(row['Total Accounts']):,}"
            gr[2].text = f"{int(row['Churned Accounts']):,}"
            gr[3].text = f"{(row['Churn Rate']*100):.1f}%"

    doc.add_heading("3. Key Retention Strategic Recommendations", level=2)
    doc.add_paragraph("• Proactive Interventions: Focus engagement efforts on accounts aged 46–60 exhibiting inactivity.", style='List Bullet')
    doc.add_paragraph("• Regional Prioritization: Address elevated attrition in the German market through tailored premium service tiers.", style='List Bullet')
    doc.add_paragraph("• Product Portfolio Optimization: Cross-sell towards 2 banking products to lock in customer loyalty.", style='List Bullet')

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()
