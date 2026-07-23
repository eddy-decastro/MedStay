# Dockerfile SIMPLE Phase 1 (~10 lignes). Multi-stage + non-root + HEALTHCHECK
# arrivent en Phase 5 (CLAUDE.md : deploiement d'abord, optimisation ensuite).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render route le trafic via $PORT (defini au runtime), EXPOSE est purement
# informatif ici et sert surtout au `docker run -p 7860:7860` en local.
EXPOSE 7860

RUN chmod +x start.sh
CMD ["./start.sh"]
