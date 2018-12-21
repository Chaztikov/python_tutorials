#Quick Start
#The basic idea is to transform a PDF document into an element tree so we can find items with JQuery-like selectors using pyquery. Suppose we're trying to extract a name from a set of PDFs, but all we know is that it appears underneath the words "Your first name and initial" in each PDF:

import pdfquery,sys,os
os.chdir('../pdfquery/')
pdf = pdfquery.PDFQuery("tests/samples/IRS_1040A.pdf")
pdf.load()
label = pdf.pq('LTTextLineHorizontal:contains("Your first name and initial")')
left_corner = float(label.attr('x0'))
bottom_corner = float(label.attr('y0'))
name = pdf.pq('LTTextLineHorizontal:in_bbox("%s, %s, %s, %s")' % (left_corner, bottom_corner-30, left_corner+150, bottom_corner)).text()
#name
#'John E.'
#Note that we don't have to know where the name is on the page, or what page it's on, or how the PDF has it stored internally.

#Performance Note: The initial call to pdf.load() runs very slowly, because the underlying pdfminer library has to compare every element on the page to every other element. See the Caching section to avoid this on subsequent runs.

#Now let's extract and format a bunch of data all at once:

pdf = pdfquery.PDFQuery("tests/samples/IRS_1040A.pdf")
pdf.extract( [
     ('with_parent','LTPage[pageid=1]'),
     ('with_formatter', 'text'),

     ('last_name', 'LTTextLineHorizontal:in_bbox("315,680,395,700")'),
     ('spouse', 'LTTextLineHorizontal:in_bbox("170,650,220,680")'),

     ('with_parent','LTPage[pageid=2]'),

     ('oath', 'LTTextLineHorizontal:contains("perjury")', lambda match: match.text()[:30]+"..."),
     ('year', 'LTTextLineHorizontal:contains("Form 1040A (")', lambda match: int(match.text()[-5:-1]))
 ])
#Result:

# {'last_name': 'Michaels',
#  'spouse': 'Susan R.',
#  'year': 2007,
#  'oath': 'Under penalties of perjury, I ...',}
