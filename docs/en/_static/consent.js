window.dataLayer = window.dataLayer || [];
function gtag(){ dataLayer.push(arguments); }

// EU/EEA・英国のみ denied
gtag('consent', 'default', {
  'ad_storage': 'denied',
  'ad_user_data': 'denied',
  'ad_personalization': 'denied',
  'analytics_storage': 'granted'
});
gtag('consent', 'default', {
  'region': ['AT','BE','BG','HR','CY','CZ','DK','EE','FI','FR','DE','GR',
             'HU','IS','IE','IT','LV','LI','LT','LU','MT','NL','NO','PL',
             'PT','RO','SK','SI','ES','SE','GB','CH'],
  'analytics_storage': 'denied'
});

gtag('js', new Date());
gtag('config', 'G-59F6KL8C5D');

// gtag.js を動的に読み込む
var s = document.createElement('script');
s.async = true;
s.src = 'https://www.googletagmanager.com/gtag/js?id=G-59F6KL8C5D';
document.head.appendChild(s);