FROM caddy

# Copy the static website
# Use the .dockerignore file to control what ends up inside the image!
COPY . .

# set caddy port
RUN echo "http://*:8000" > /etc/caddy/Caddyfile
RUN echo "rewrite /repl /repl.html" >> /etc/caddy/Caddyfile
RUN echo "rewrite /compilerepl /compilerepl.html" >> /etc/caddy/Caddyfile
RUN echo "file_server" >> /etc/caddy/Caddyfile
