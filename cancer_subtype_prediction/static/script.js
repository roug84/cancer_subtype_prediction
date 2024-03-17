// custom_script.js
$(document).ready(function() {
    // Initialize DataTables
    {% for idx in range(df_htmls|length) %}
    $('#dataframe-{{ idx }}').DataTable({
        "scrollX": true,
        "autoWidth": false
    });
    {% endfor %}

    // Initialize Swiper
    var swiper = new Swiper('.swiper-container', {
        slidesPerView: 1,
        spaceBetween: 30,
        pagination: {
            el: '.swiper-pagination',
            clickable: true,
        },
        navigation: {
            nextEl: '.swiper-button-next',
            prevEl: '.swiper-button-prev',
        },
        on: {
            slideChange: function() {
                if (this.activeIndex === 1) { // Adjust the index based on where the SHAP plot is
                    renderShapPlot(); // Call the function to render the SHAP plot
                } else if (this.activeIndex === 3) { // Adjust the index based on where the Correlation plot is
                    renderCorrelationPlot(); // Call the function to render the Correlation plot
                }
            },
        },
    });

    // Function to render the SHAP plot
    function renderShapPlot() {
        var shapFigureData = SHAP_PLOT_DATA; // Placeholder for SHAP plot data
        if (shapFigureData) {
            Plotly.newPlot('shap-plot-placeholder', shapFigureData.data, shapFigureData.layout);
        }
    }

    // Function to render the Correlation plot
    function renderCorrelationPlot() {
        var correlationData = CORRELATION_PLOT_DATA; // Placeholder for correlation plot data
        if (correlationData) {
            var plotData = [{
                type: 'scatter',
                mode: 'markers', // Use 'markers' mode for scatter plot
                x: correlationData.x,
                y: correlationData.y,
                marker: {
                    color: 'blue', // Customize marker color
                    size: 10 // Customize marker size
                }
            }];

            var layout = {
                title: correlationData.title,
                xaxis: { title: correlationData.xaxis },  // Customize the axis titles as needed
                yaxis: { title: correlationData.yaxis }
            };

            Plotly.newPlot('correlation-plot', plotData, layout);
        }
    }
});
