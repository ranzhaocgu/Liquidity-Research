figure
hold on
plot(put_option_strike_bbg,put_option_vol_sim);
clear title;
title('3-month Implied Volatility for Call Options');
xlabel('strike/moneyness');
ylabel('vol');
plot(put_option_strike_bbg,put_option_vol_bbg);
legend('Simulation', 'Bloomberg','Location','NorthEast')
hold off