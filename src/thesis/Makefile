LATEX_PDF=pdflatex
LOPTS=-halt-on-error -interaction=batchmode

M_TEMPLATE_GENERATED_TEX=Template_MastersThesis.tex Sablona_DP_UTF-8.tex Sablona_sk_DP_UTF-8.tex
B_TEMPLATE_ICONVERTED_TEX=Sablona_BP_ISO-8859-2.tex Sablona_BP_Windows-1250.tex Sablona_sk_BP_ISO-8859-2.tex Sablona_sk_BP_Windows-1250.tex
M_TEMPLATE_ICONVERTED_TEX=Sablona_DP_ISO-8859-2.tex Sablona_DP_Windows-1250.tex Sablona_sk_DP_ISO-8859-2.tex Sablona_sk_DP_Windows-1250.tex
TEMPLATE_GENERATED_TEX=$(B_TEMPLATE_ICONVERTED_TEX) $(M_TEMPLATE_ICONVERTED_TEX) $(M_TEMPLATE_GENERATED_TEX)

M_TEMPLATE_GENERATED_PDF=$(M_TEMPLATE_GENERATED_TEX:.tex=.pdf)
B_TEMPLATE_ICONVERTED_PDF=$(B_TEMPLATE_ICONVERTED_TEX:.tex=.pdf)
M_TEMPLATE_ICONVERTED_PDF=$(M_TEMPLATE_ICONVERTED_TEX:.tex=.pdf)
TEMPLATE_GENERATED_PDF=$(TEMPLATE_GENERATED_TEX:.tex=.pdf)

JUSTCOMPILE_TEMPLATE_TEX=Template_BachelorsThesis.tex Sablona_BP_UTF-8.tex MastersThesis_example.tex  DP_Priklad.tex Sablona_sk_BP_UTF-8.tex
JUSTCOMPILE_TEMPLATE_PDF=$(JUSTCOMPILE_TEMPLATE_TEX:.tex=.pdf)

JUSTCOMPILE_COVER_TEX=deskyBP_UTF-8_cernobile.tex coverBachelors-bw.tex
JUSTCOMPILE_COVER_PDF=$(JUSTCOMPILE_COVER_TEX:.tex=.pdf)

COVER_GENERATED_BASIC_TEX=coverBachelors-inverse_colours.tex coverMasters-bw.tex coverMasters-inverse_colours.tex deskyBP_UTF-8_invert-barvy.tex deskyDP_UTF-8_cernobile.tex deskyDP_UTF-8_invert-barvy.tex
COVER_GENERATED_BASIC_PDF=$(COVER_GENERATED_BASIC_TEX:.tex=.pdf)

COVER_GENERADED_ICONV_TEX=deskyBP_ISO-8859-2_cernobile.tex deskyBP_Windows-1250_cernobile.tex deskyDP_ISO-8859-2_cernobile.tex deskyDP_Windows-1250_cernobile.tex deskyBP_ISO-8859-2_invert-barvy.tex deskyBP_Windows-1250_invert-barvy.tex deskyDP_ISO-8859-2_invert-barvy.tex deskyDP_Windows-1250_invert-barvy.tex

COVER_IMAGES=cvut-logo-w.pdf cvut-logo-bw-symbol.pdf

TEMPLATE_TEX= $(JUSTCOMPILE_TEMPLATE_TEX) $(TEMPLATE_GENERATED_TEX)
TEMPLATE_PDF= $(TEMPLATE_TEX:.tex=.pdf)

TEMPLATE_README_EN_TXT=README_t.txt
TEMPLATE_README_EN_PDF=README_t.pdf
TEMPLATE_README_EN=$(TEMPLATE_README_EN_TXT) $(TEMPLATE_README_EN_PDF)

TEMPLATE_DECLARATION_CS=prohlaseni1.tex.txt prohlaseni2.tex.txt prohlaseni3.tex.txt prohlaseni4.tex.txt prohlaseni5.tex.txt prohlaseni6.tex.txt

TEMPLATE_README_CS_TXT=CtiMe_s.txt
TEMPLATE_README_CS_PDF=CtiMe_s.pdf
TEMPLATE_README_CS=$(TEMPLATE_README_CS_TXT) $(TEMPLATE_README_CS_PDF)

COVER_README_EN_TXT=README_c.txt
COVER_README_EN_PDF=README_c.pdf
COVER_README_EN=$(COVER_README_EN_PDF)

COVER_README_CS_TXT=CtiMe_d.txt
COVER_README_CS_PDF=CtiMe_d.pdf
COVER_README_CS=$(COVER_README_CS_PDF)

TEMPLATE_ZIP=$(TEMPLATE_DECLARATION_CS) $(JUSTCOMPILE_TEMPLATE_TEX) FITthesis.cls iso690.bst csn690.bst mybibliographyfile.bib ref.bib cvut-logo-bw.pdf cvut-logo-bw-en.pdf gnuplot-bw.pdf gnuplot-col.pdf DP_Priklad.pdf MastersThesis_example.pdf $(TEMPLATE_GENERATED_TEX) $(TEMPLATE_README_EN) $(TEMPLATE_README_CS)

COVER_ZIP=$(JUSTCOMPILE_COVER_TEX) FITcover.cls $(JUSTCOMPILE_COVER_PDF) $(COVER_GENERATED_BASIC_TEX) $(COVER_GENERATED_BASIC_PDF) $(COVER_GENERADED_ICONV_TEX) $(COVER_IMAGES) $(COVER_README_EN) $(COVER_README_CS)


all:	$(JUSTCOMPILE_TEMPLATE_TEX) $(JUSTCOMPILE_COVER_TEX) FITthesis-template.zip FITthesis-cover.zip

FITthesis-template.zip:	$(TEMPLATE_ZIP) $(TEMPLATE_PDF)
	zip -q -9 FITthesis-template.zip $(TEMPLATE_ZIP)
	rm --force *.log $(TEMPLATE_GENERATED_TEX) $(TEMPLATE_PDF) $(TEMPLATE_README_CS) $(TEMPLATE_README_EN) CtiMe-*

FITthesis-cover.zip:$(COVER_ZIP)
	zip -q -9 FITthesis-cover.zip $(COVER_ZIP)

%.bbl : %.tex %.aux iso690.bst mybibliographyfile.bib
	- bibtex -terse $*

%.acr : %.tex %.aux
	makeglossaries -q $*

%.aux : %.tex
	$(LATEX_PDF) $(LOPTS) $*

prohlaseni%.tex.txt : prohlaseni%.utf8.txt
	cstocs utf8 tex $< > $@

DP_Priklad.pdf : DP_Priklad.tex csn690.bst mybibliographyfile.bib DP_Priklad.bbl

MastersThesis_example.pdf : MastersThesis_example.tex iso690.bst mybibliographyfile.bib MastersThesis_example.bbl #MastersThesis_example.acr

%.pdf : %.aux
	$(LATEX_PDF) $(LOPTS) $*
	$(LATEX_PDF) $(LOPTS) $*
	rm --force $*.aux $*.bbl $*.blg $*.lof $*.lot $*.out $*.toc

%.dvi : %.tex
	latex $(LOPTS) $*
	rm --force $*.aux

%-ascii.tex : %.tex
	cstocs utf8 ascii $< > $@

%.txt : %-ascii.dvi
	dvi2tty -q -u -c -o $@ $<

%.txt : %.dvi
	dvi2tty -q -u -c -o $@ $<

Template_MastersThesis.tex : Template_BachelorsThesis.tex
	sed 's/\[thesis=B/\[thesis=M/' $< > $@

coverMasters-bw.tex : coverBachelors-bw.tex
	sed 's/\[thesis=B/\[thesis=M/' $< > $@

%_DP_UTF-8.tex : %_BP_UTF-8.tex
	sed 's/\[thesis=B/\[thesis=M/' $< > $@

deskyDP_UTF-8_cernobile.tex : deskyBP_UTF-8_cernobile.tex
	sed 's/\[thesis=B/\[thesis=M/' $< > $@

%-inverse_colours.tex : %-bw.tex
	sed 's/,english]{FITcover}/,english,inverse]{FITcover}/' $< > $@

%_invert-barvy.tex : %_cernobile.tex
	sed 's/,czech]{FITcover}/,czech,inverse]{FITcover}/' $< > $@

%P_ISO-8859-2.tex : %P_UTF-8.tex
	sed 's/usepackage\[utf8\]/usepackage\[latin2\]/' $< | sed 's/LaTeX source encoded as UTF-8/LaTeX source encoded as ISO-8859-2/' | iconv --from-code=UTF-8 --to-code=ISO-8859-2 --output=$@
	
%P_Windows-1250.tex : %P_UTF-8.tex
	sed 's/usepackage\[utf8\]/usepackage\[cp1250\]/' $< | sed 's/LaTeX source encoded as UTF-8/LaTeX source encoded as Windows-1250/' | iconv --from-code=UTF-8 --to-code=Windows-1250 --output=$@

%P_ISO-8859-2_cernobile.tex : %P_UTF-8_cernobile.tex
	sed 's/usepackage\[utf8\]/usepackage\[latin2\]/' $< | sed 's/LaTeX source encoded as UTF-8/LaTeX source encoded as ISO-8859-2/' | iconv --from-code=UTF-8 --to-code=ISO-8859-2 --output=$@
	
%P_Windows-1250_cernobile.tex : %P_UTF-8_cernobile.tex
	sed 's/usepackage\[utf8\]/usepackage\[cp1250\]/' $< | sed 's/LaTeX source encoded as UTF-8/LaTeX source encoded as Windows-1250/' | iconv --from-code=UTF-8 --to-code=Windows-1250 --output=$@


clean:
	-rm --force *.bbl *.blg *.out *.djs *.aux *.log *.zip *.lof *.lot *.toc *.acn *.acr *.alg *.gls *.ist *.glo *.glg $(TEMPLATE_GENERATED_TEX) $(TEMPLATE_PDF) $(JUSTCOMPILE_COVER_PDF) $(COVER_GENERATED_BASIC_TEX) $(COVER_GENERATED_BASIC_PDF) $(COVER_GENERADED_ICONV_TEX) $(TEMPLATE_README_CS) $(TEMPLATE_README_EN) $(COVER_README_CS) $(COVER_README_EN) CtiMe-*
