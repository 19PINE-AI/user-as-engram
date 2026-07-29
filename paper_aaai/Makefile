DOCUMENT := AnonymousSubmission2027

LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -halt-on-error -file-line-error

TEX_SOURCES := $(wildcard *.tex)
BIB_SOURCES := $(wildcard *.bib)
STYLE_SOURCES := $(wildcard *.sty *.bst)
FIGURES := $(wildcard figs/*.pdf figs/*.png figs/*.jpg figs/*.jpeg)

.DEFAULT_GOAL := all

.PHONY: all clean distclean watch

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(TEX_SOURCES) $(BIB_SOURCES) $(STYLE_SOURCES) $(FIGURES)
	$(LATEXMK) $(LATEXMK_FLAGS) $(DOCUMENT).tex

watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(DOCUMENT).tex

clean:
	$(LATEXMK) -c $(DOCUMENT).tex

distclean:
	$(LATEXMK) -C $(DOCUMENT).tex
