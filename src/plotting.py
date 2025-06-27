 # Toda a lógica de criação de gráficos


# src/plotting.py
import pandas as pd
import matplotlib.pyplot as plt

def create_plot(df, chart_type, col_x, col_y):
    """Gera uma figura Matplotlib a partir do DataFrame."""
    if not col_x or col_x not in df.columns or not col_y or col_y not in df.columns:
        raise ValueError("Colunas selecionadas para o gráfico são inválidas.")

    fig, ax = plt.subplots(figsize=(10, 6))

    is_col_x_numeric = pd.api.types.is_numeric_dtype(df[col_x])
    is_col_y_numeric = pd.api.types.is_numeric_dtype(df[col_y])

    try:
        # Lógica de geração de gráfico
        if chart_type == 'Automático':
            if is_col_x_numeric and is_col_y_numeric:
                chart_type = 'Dispersão'
            elif not is_col_x_numeric and is_col_y_numeric:
                 chart_type = 'Barras'
            elif is_col_x_numeric and not is_col_y_numeric:
                 chart_type = 'Barras'
                 col_x, col_y = col_y, col_x # Inverte para ter categoria no eixo X
            else:
                raise ValueError("Não foi possível gerar um gráfico automático com as colunas fornecidas.")

        if chart_type == 'Dispersão':
            if not is_col_x_numeric or not is_col_y_numeric:
                raise ValueError('Dispersão requer duas colunas numéricas.')
            ax.scatter(df[col_x], df[col_y], color='teal', alpha=0.6)
            ax.set_title(f'Dispersão entre {col_x} e {col_y}', fontsize=14)

        elif chart_type in ['Barras', 'Linha', 'Pizza']:
            if is_col_x_numeric == is_col_y_numeric:
                raise ValueError(f'Gráfico de {chart_type} requer uma coluna categórica e uma numérica.')
            
            col_cat, col_val = (col_x, col_y) if not pd.api.types.is_numeric_dtype(df[col_x]) else (col_y, col_x)
            df_grouped = df.groupby(col_cat)[col_val].sum().sort_values(ascending=False)

            if chart_type == 'Barras':
                df_grouped.head(15).plot(kind='bar', ax=ax, color='cornflowerblue')
                ax.set_title(f'Total de {col_val} por {col_cat}')
                plt.xticks(rotation=45, ha='right')
            
            elif chart_type == 'Linha':
                df_grouped.head(20).sort_index().plot(kind='line', marker='o', color='green', ax=ax)
                ax.set_title(f'Tendência de {col_val} por {col_cat}')
                plt.xticks(rotation=45, ha='right')

            elif chart_type == 'Pizza':
                df_pie = df_grouped.head(7)
                if len(df_grouped) > 7:
                    df_pie['Outros'] = df_grouped[7:].sum()
                
                ax = df_pie.plot(kind='pie', autopct='%1.1f%%', startangle=90, legend=False, ax=ax)
                ax.set_ylabel('')
                ax.set_title(f'Distribuição de {col_val} por {col_cat}')
                plt.legend(labels=df_pie.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        ax.set_xlabel(col_x, fontsize=12)
        ax.set_ylabel(col_y, fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        return fig

    except Exception as e:
        plt.close(fig) # Garante que a figura seja fechada em caso de erro
        raise e