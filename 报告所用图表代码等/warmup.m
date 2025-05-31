% 定义数据
warmup_ratios = [0.05, 0.1, 0.15];
accuracy = [0.8420, 0.8642, 0.8469];
f1_scores = [0.8232, 0.8501, 0.8315];

% 创建分组条形图
figure;
bar_data = [accuracy; f1_scores]';  % 转置为3行2列
hBar = bar(bar_data, 'grouped');
set(gca, 'XTickLabel', {'0.05', '0.10', '0.15'});

% 设置图形属性
title('模型性能随预热比例的变化');
xlabel('预热比例 (warmup\_ratio)');
ylabel('分数');
ylim([0.8 0.88]);  % 聚焦在差异明显的范围
grid on;

% 添加条形图数值标签
for i = 1:length(hBar)
    xData = hBar(i).XEndPoints;
    yData = hBar(i).YEndPoints;
    text(xData, yData, num2str(yData', '%.4f'), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 9);
end

% 添加图例
legend({'准确率', 'F1分数'}, 'Location', 'northwest');