# Set things up
FROM alpine:latest as build
ARG VER=3.0.2
ARG COSMO=cosmos-$VER.zip
RUN wget https://github.com/jart/cosmopolitan/releases/download/$VER/$COSMO
WORKDIR cosmo
RUN unzip ../$COSMO bin/ape.elf bin/assimilate bin/bash bin/python bin/zip
RUN mkdir Lib
COPY scrapscript.py Lib
RUN sh bin/zip -A -r bin/python Lib
RUN bin/ape.elf bin/assimilate bin/python

# Set up the container
FROM scratch
COPY --from=build /cosmo/bin/python .
EXPOSE 8000
ENTRYPOINT ["./python", "-m", "scrapscript"]
CMD ["serve", "--port", "8000"]
