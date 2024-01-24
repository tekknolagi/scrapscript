# Set things up
FROM alpine:latest as build
ARG VER=3.0.2
ARG COSMO=cosmos-$VER.zip
RUN wget https://github.com/jart/cosmopolitan/releases/download/$VER/$COSMO
WORKDIR cosmo
RUN unzip ../$COSMO bin/ape.elf bin/assimilate bin/bash bin/python bin/zip
# Remove some packages we're never going to use
RUN sh bin/zip -A --delete bin/python "Lib/site-packages/*"
RUN mkdir Lib
COPY scrapscript.py Lib
COPY .args .
RUN bin/ape.elf bin/python -m compileall Lib
RUN mv Lib/__pycache__/scrapscript*.pyc Lib/scrapscript.pyc
RUN rm Lib/scrapscript.py
RUN cp bin/python bin/scrapscript.com
COPY style.css Lib
COPY repl.html Lib
RUN sh bin/zip -A -r bin/scrapscript.com Lib .args
RUN bin/ape.elf bin/assimilate bin/scrapscript.com

# Set up the container
FROM scratch
COPY --from=build /cosmo/bin/scrapscript.com .
EXPOSE 8000
ENTRYPOINT ["./scrapscript.com"]
CMD ["repl"]
