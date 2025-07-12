import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display
from transformers import AutoTokenizer
import html as html_module
from fire import Fire
import pandas as pd
from tqdm import tqdm
from loguru import logger

def visualize_token_logprobs(tokens, logprob_list, tokenizer_name="Qwen/Qwen3-32B", method="html"):
    """
    Visualize text with tokens colored by their log probabilities.

    Args:
        string: The generated text string
        logprob_list: List of log probabilities for each token
        tokenizer_name: HuggingFace tokenizer name
        method: "html" or "matplotlib" for visualization method
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_strings = [tokenizer.decode([token]) for token in tokens]

    # Ensure logprob_list matches token count
    if len(logprob_list) != len(tokens):
        logger.info(f"Warning: logprob_list length ({len(logprob_list)}) doesn't match token count ({len(tokens)})")
        # Truncate or pad as needed
        if len(logprob_list) > len(tokens):
            logprob_list = logprob_list[:len(tokens)]
        else:
            logprob_list = logprob_list + [min(logprob_list)] * (len(tokens) - len(logprob_list))

    # Normalize logprobs to [0, 1] range for color mapping
    logprobs_array = np.array(logprob_list)
    min_logprob = np.min(logprobs_array)
    max_logprob = np.max(logprobs_array)

    if max_logprob - min_logprob > 0:
        normalized_logprobs = (logprobs_array - min_logprob) / (max_logprob - min_logprob)
    else:
        normalized_logprobs = np.ones_like(logprobs_array) * 0.5

    if method == "html":
        return create_html_visualization(token_strings, normalized_logprobs, logprob_list)
    else:
        return create_matplotlib_visualization(token_strings, normalized_logprobs, logprob_list)


def create_html_visualization(token_strings, normalized_logprobs, original_logprobs):
    """Create an HTML visualization with colored tokens."""
    html_parts = ['<div style="font-family: monospace; font-size: 14px; line-height: 1.8;">']

    for token, norm_prob, orig_prob in zip(token_strings, normalized_logprobs, original_logprobs):
        # Create color from light red to dark red
        # Light red: rgb(255, 200, 200), Dark red: rgb(139, 0, 0)
        r = int(255 - (255 - 139) * norm_prob)
        g = int(200 - 200 * norm_prob)
        b = int(200 - 200 * norm_prob)

        # Escape HTML characters in token
        escaped_token = html_module.escape(token)

        # Create span with color and tooltip
        span = f'<span style="background-color: rgb({r}, {g}, {b}); color: white; ' \
               f'padding: 2px 1px; margin: 0 1px; border-radius: 3px; ' \
               f'display: inline-block;" title="logprob: {orig_prob:.4f}">' \
               f'{escaped_token}</span>'

        html_parts.append(span)

    html_parts.append('</div>')

    # Add legend
    html_parts.append(create_color_legend())

    return ''.join(html_parts)


def create_color_legend():
    """Create a color legend for the visualization."""
    legend_html = '''
    <div style="margin-top: 20px; font-family: Arial, sans-serif;">
        <h4>Log Probability Scale</h4>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <span style="margin-right: 10px;">Low</span>
            <div style="width: 300px; height: 20px; background: linear-gradient(to right, 
                rgb(255, 200, 200) 0%, rgb(139, 0, 0) 100%); border: 1px solid #ccc;"></div>
            <span style="margin-left: 10px;">High</span>
        </div>
    </div>
    '''
    return legend_html


def create_matplotlib_visualization(token_strings, normalized_logprobs, original_logprobs):
    """Create a matplotlib visualization with colored tokens."""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8),
                                   gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.3})

    # Create colormap from light red to dark red
    colors = ['#FFC8C8', '#8B0000']  # Light red to dark red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('logprob', colors, N=n_bins)

    # Calculate positions for tokens
    x_pos = 0
    y_pos = 0
    line_height = 0.1
    max_width = 0.95

    positions = []
    for i, token in enumerate(token_strings):
        # Estimate token width (rough approximation)
        token_width = len(token) * 0.01

        # Check if we need to wrap to next line
        if x_pos + token_width > max_width:
            x_pos = 0
            y_pos -= line_height

        positions.append((x_pos, y_pos))
        x_pos += token_width + 0.005

    # Plot tokens
    for i, (token, pos, norm_prob) in enumerate(zip(token_strings, positions, normalized_logprobs)):
        color = cmap(norm_prob)
        ax1.text(pos[0], pos[1], token,
                 color='black',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=color, edgecolor='none'),
                 fontsize=10,
                 family='monospace',
                 verticalalignment='center')

    # Set up main plot
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(min(-0.2, min(p[1] for p in positions) - 0.1), 0.1)
    ax1.axis('off')
    ax1.set_title('Token Log Probability Visualization', fontsize=16, pad=20)

    # Create colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=ax2, orientation='horizontal')
    cbar.set_label('Log Probability (normalized)', fontsize=12)

    # Add min/max labels
    min_logprob = min(original_logprobs)
    max_logprob = max(original_logprobs)
    ax2.text(0, -0.5, f'Min: {min_logprob:.4f}', transform=ax2.transAxes, fontsize=10)
    ax2.text(1, -0.5, f'Max: {max_logprob:.4f}', transform=ax2.transAxes,
             fontsize=10, ha='right')

    plt.tight_layout()
    return fig


# Example usage function
def example_usage():
    """Example of how to use the visualization functions."""
    # Example data
    example_string = "The model generates text based on probability distributions."

    # Simulated log probabilities (replace with your actual data)
    example_logprobs = [-2.5, -0.8, -1.2, -3.0, -0.5, -1.8, -2.2, -0.3, -1.5]

    # Create HTML visualization
    html_viz = visualize_token_logprobs(
        example_string,
        example_logprobs,
        tokenizer_name="Qwen/Qwen3-32B",
        method="html"
    )

    # Display in Jupyter notebook
    display(HTML(html_viz))

    # Create matplotlib visualization
    fig = visualize_token_logprobs(
        example_string,
        example_logprobs,
        tokenizer_name="Qwen/Qwen3-32B",
        method="matplotlib"
    )
    plt.show()


# Advanced visualization with token boundaries
def visualize_with_token_info(tokens, logprob_list, tokenizer_name="Qwen/Qwen3-32B"):
    """
    Enhanced visualization showing token boundaries and detailed information.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    token_strings = [tokenizer.decode([token]) for token in tokens]

    # Create detailed HTML with token information
    html_parts = ['''
    <style>
        .token-viz { 
            font-family: 'Courier New', monospace; 
            font-size: 14px; 
            line-height: 2.5; 
            margin: 20px;
        }
        .token {
            color: white;
            display: inline-block;
            padding: 4px 6px;
            margin: 2px;
            border-radius: 4px;
            border: 1px solid rgba(0,0,0,0.1);
            cursor: pointer;
            position: relative;
        }
        .token:hover {
            border-color: #333;
            transform: scale(1.05);
            transition: all 0.2s;
        }
        .token-info {
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: #333;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s;
        }
        .token:hover .token-info {
            opacity: 1;
        }
        .stats {
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            font-family: Arial, sans-serif;
        }
    </style>
    <div class="token-viz">
    ''']

    # Normalize logprobs
    logprobs_array = np.array(logprob_list[:len(tokens)])
    min_logprob = np.min(logprobs_array)
    max_logprob = np.max(logprobs_array)

    if max_logprob - min_logprob > 0:
        normalized_logprobs = (logprobs_array - min_logprob) / (max_logprob - min_logprob)
    else:
        normalized_logprobs = np.ones_like(logprobs_array) * 0.5

    # Create token spans
    for i, (token, norm_prob, orig_prob) in enumerate(zip(token_strings, normalized_logprobs, logprobs_array)):
        # Color calculation
        r = int(255 - (255 - 139) * norm_prob)
        g = int(200 - 200 * norm_prob)
        b = int(200 - 200 * norm_prob)

        escaped_token = html_module.escape(token).replace(' ', '&nbsp;')

        html_parts.append(f'''
        <span class="token" style="background-color: rgb({r}, {g}, {b});">
            {escaped_token}
            <span class="token-info">
                Token #{i} | ID: {tokens[i]} | LogProb: {orig_prob:.4f}
            </span>
        </span>
        ''')

    html_parts.append('</div>')

    # Add statistics
    html_parts.append(f'''
    <div class="stats">
        <h4>Token Statistics</h4>
        <p><strong>Total tokens:</strong> {len(tokens)}</p>
        <p><strong>Average log probability:</strong> {np.mean(logprobs_array):.4f}</p>
        <p><strong>Min log probability:</strong> {min_logprob:.4f}</p>
        <p><strong>Max log probability:</strong> {max_logprob:.4f}</p>
        <p><strong>Std deviation:</strong> {np.std(logprobs_array):.4f}</p>
    </div>
    ''')

    # Add color legend
    html_parts.append(create_color_legend())

    return ''.join(html_parts)


# Utility function to save visualization
def save_visualization(html_content, filename="token_visualization.html"):
    """Save the HTML visualization to a file."""
    full_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Token Log Probability Visualization</title>
        <meta charset="utf-8">
    </head>
    <body>
        {html_content}
    </body>
    </html>
    '''

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(full_html)
    logger.info(f"Visualization saved to {filename}")


def main_single_sample(df, sample_id: int):
    metric = next(iter(df.metric[sample_id].values()))
    model_response = df.model_response[sample_id]

    tokens = model_response["output_tokens"][0]
    logprob_list = []
    for logprob_dict in tqdm(model_response["logprobs"][0], desc=f"Extracting logprobs for sample {sample_id}"):
        for key, value in logprob_dict.items():
            if value is not None and value["rank"] == 1:
                logprob_list.append(value["logprob"])

    assert len(tokens) == len(logprob_list)
    logger.info(f"Sample ID: {sample_id} with {len(tokens)} response tokens (metric: {metric})")

    # plot!
    enhanced_html = visualize_with_token_info(tokens, logprob_list)
    save_visualization(enhanced_html, f"plots/token-logprobs-{sample_id}.html")


def main(detail_file_path: str, sample_id: int = None):
    df = pd.read_parquet(detail_file_path)

    if sample_id is None:
        num_samples = len(df.metric)
        logger.info(f"Default plotting all samples ({num_samples})")
        for idx in range(num_samples):
            main_single_sample(df, idx)
    else:
        main_single_sample(df, sample_id)


if __name__ == "__main__":
    Fire(main)
