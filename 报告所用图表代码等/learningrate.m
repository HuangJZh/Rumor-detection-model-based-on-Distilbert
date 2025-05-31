% 定义数据
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5];
accuracy = [0.7778, 0.7975, 0.8222, 0.8321, 0.8494, 0.8519, 0.8346];
f1_scores = [0.7500, 0.7697, 0.7978, 0.8046, 0.8262, 0.8256, 0.8069];

% 创建图形窗口
figure('Color', 'white', 'Position', [100, 100, 800, 600]);

% 绘制准确率曲线
plot(learning_rates, accuracy, 'b-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;

% 绘制F1分数曲线
plot(learning_rates, f1_scores, 'r-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'r');

% 添加数据标签
for i = 1:length(learning_rates)
    text(learning_rates(i), accuracy(i)+0.005, sprintf('%.4f', accuracy(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    text(learning_rates(i), f1_scores(i)-0.008, sprintf('%.4f', f1_scores(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.8 0 0]);
end

% 设置图形属性
title('模型性能 vs 学习率', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('学习率', 'FontSize', 12);
ylabel('性能指标', 'FontSize', 12);
legend({'准确率', 'F1分数'}, 'Location', 'northwest');
grid on;
box on;

% 设置坐标轴范围
xlim([0.8e-5, 7.2e-5]);
ylim([0.73, 0.87]);

% 设置x轴为科学计数法
ax = gca;
ax.XAxis.Exponent = -5;
ax.XAxis.TickLabelFormat = '%.0e';

% 添加峰值标记
[~, max_acc_idx] = max(accuracy);
line([learning_rates(max_acc_idx), learning_rates(max_acc_idx)], [0.73, accuracy(max_acc_idx)], ...
    'Color', [0.5 0.5 0.5], 'LineStyle', '--', 'LineWidth', 1.5);
text(learning_rates(max_acc_idx), 0.735, sprintf('峰值: %.4f', accuracy(max_acc_idx)), ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'BackgroundColor', 'white');