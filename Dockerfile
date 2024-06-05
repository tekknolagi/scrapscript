FROM caddy
COPY . .
RUN echo ":8000" > /etc/caddy/Caddyfile
RUN echo "rewrite /repl /repl.html" >> /etc/caddy/Caddyfile
RUN echo "rewrite /compilerepl /compilerepl.html" >> /etc/caddy/Caddyfile
RUN echo "log" >> /etc/caddy/Caddyfile
RUN echo "file_server" >> /etc/caddy/Caddyfile
