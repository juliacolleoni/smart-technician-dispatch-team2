# Otimização Global por Score Combinado

## Mudança Implementada

### ANTES (otimização individual de features):
1. **Fase 1**: Escolhia técnico baseado **apenas em skill_score**
2. **Fase 2**: Otimizava rota por distância (KNN)
3. **Fase 3**: Agendava horários por disponibilidade
4. **Score combinado**: Era calculado mas NÃO usado na tomada de decisão

### DEPOIS (otimização global):
1. **Fase 1**: Escolhe técnico baseado no **SCORE COMBINADO COMPLETO**
   - `score = 0.35 × skill + 0.30 × availability + 0.35 × travel`
   - Calcula skill, availability E travel para CADA combinação técnico-trabalho
   - Travel score considera:
     - Se o técnico já tem jobs naquele dia: distância ao job mais próximo
     - Se é o primeiro job do dia: distância da base do técnico
2. **Fase 2**: Mantém otimização de rota por distância (KNN)
3. **Fase 3**: Mantém agendamento por disponibilidade

## Resultados

### Estatísticas dos Scores:
- **Score médio**: 0.541
- **Score mediano**: 0.520
- **Range**: 0.310 - 0.900

### Contribuição Média por Componente:
- **Skill**: 0.493 → 0.172 (peso 0.35)
- **Availability**: 0.485 → 0.145 (peso 0.30)
- **Travel**: 0.643 → 0.225 (peso 0.35)

### Trade-off:
- Distância total aumentou 2.8% (69.98 km)
- Mas agora temos MELHOR balanço entre skill, disponibilidade e distância
- O algoritmo escolhe técnicos com melhor score global, não apenas melhor skill

## Vantagens

1. ✅ **Otimização verdadeiramente global**: Considera todos os fatores simultaneamente
2. ✅ **Decisões mais balanceadas**: Pode escolher técnico com skill 0.8 mas travel 0.9 ao invés de skill 0.85 mas travel 0.3
3. ✅ **Score reflete realidade**: O score salvo no output é o mesmo usado na decisão
4. ✅ **Considera contexto**: Travel score leva em conta os jobs já atribuídos ao técnico naquele dia

## Código Modificado

**Arquivo**: `technician_dispatch_optimizer.py`
**Linhas**: ~510-595

Principais mudanças:
- Cálculo de `combined_score` para cada técnico candidato
- Decisão baseada em `if combined_score > best_combined_score` ao invés de `if skill_score > best_skill_score`
- Cálculo dinâmico de travel_score baseado em jobs já atribuídos
