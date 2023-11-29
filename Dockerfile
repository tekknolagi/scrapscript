# Set things up
FROM alpine:latest as build
ARG VER=3.0.2
ARG COSMO=cosmos-$VER.zip
RUN wget https://github.com/jart/cosmopolitan/releases/download/$VER/$COSMO
WORKDIR cosmo
RUN unzip ../$COSMO
# RUN chmod +x bin/python
# RUN chmod +x bin/bash
# RUN sh bin/bash --assimilate
RUN bin/ape.elf bin/assimilate bin/bash
RUN bin/ape.elf bin/assimilate bin/python

# Set up the container
FROM alpine:latest as prod
# TODO(max): Do from scratch and use all of Justine's tools instead of
# container's.
# FROM scratch
COPY scrapscript.py .
# This copies everything under cosmo into / (not producing a cosmo directory).
# COPY --from=build /cosmo/ /
COPY --from=build /cosmo /cosmo
EXPOSE 8000
ENTRYPOINT ["/cosmo/bin/python", "scrapscript.py"]
CMD ["serve", "--port", "8000"]
