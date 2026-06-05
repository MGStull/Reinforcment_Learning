import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

#Most useful for RL research

fig = px.line(df, x='episode', y='reward', title='Training Curve')

fig.show()

fig = go.Figure(data=go.Heatmap(
    z=value_matrix,
    x=dealer_cards,
    y=player_sums,
    colorscale='RdYlGn'
))
fig.show()

#3. 3D surface --value functions 
fig = go.Figure(data=go.Surface(z=value_matrix))
fig.show

# 4. Scatter — policy visualization
fig = px.scatter(df, x='state_x', y='state_y',
                 color='action', symbol='action')
fig.show()

# 5. Subplots — multiple charts together
fig = make_subplots(rows=2, cols=2)
fig.add_trace(go.Line(y=rewards), row=1, col=1)
fig.add_trace(go.Heatmap(z=V), row=1, col=2)
fig.show()