"""
inference_week7.py - Hàm inference Tầng 3 (Xiaomi) cho Backend.

Điểm vào duy nhất:  predict_anomaly_week7(user_id, daily_history, age=None, package_dir=PACKAGE_DIR)

Quy ước (xem inference_spec_week7.md):
  - Hàm KHÔNG giữ trạng thái. BE truyền toàn bộ lịch sử ngày của user (thô, CHƯA z-score).
  - Ngày cuối cùng (theo `date`) trong daily_history là ngày cần đánh giá ("hôm nay").
  - 3 tier chọn theo số NGÀY HỢP LỆ của user (ngày 1-3 / 4-13 / >=14), mọi tier đều chạy luật cứng.
  - Tier 3: LSTM-AE OR Isolation Forest (hybrid OR); giải thích bằng sai số tái tạo theo từng feature.
"""
import json
import os

import numpy as np

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
LEVEL = {'NONE': 0, 'SOFT': 1, 'CRITICAL': 2}


class Predictor:
    def __init__(self, package_dir=PACKAGE_DIR, config=None, population=None, names_vi=None,
                 iso_forest=None, recon_fn=None):
        """recon_fn(window[7,F] float32) -> reconstruction[7,F]. Chỉ truyền khi muốn thay model (test)."""
        self.dir = package_dir
        self.cfg = config or json.load(open(os.path.join(package_dir, 'config_week7.json'), encoding='utf-8'))
        c = self.cfg
        self.features = list(c['common_features'])
        self.W = int(c['window_size'])
        self.cs = c['cold_start']
        self.clin = c['clinical']
        self.pop = population or json.load(open(os.path.join(package_dir, c['cold_start']['population_baseline_file']),
                                                encoding='utf-8'))
        self.names_vi = names_vi or json.load(open(os.path.join(package_dir, c['feature_names_file']),
                                                   encoding='utf-8'))
        self.lstm_th = float(c['lstm_ae']['threshold_value'])
        self.iso_th = float(c['isolation_forest']['score_threshold'])
        if iso_forest is None:
            import joblib
            iso_forest = joblib.load(os.path.join(package_dir, c['isolation_forest']['file']))
        self.iso = iso_forest
        self.recon_fn = recon_fn or self._load_lstm()

    # ------------------------------------------------------------------ model
    def _load_lstm(self):
        import torch
        import torch.nn as nn
        la = self.cfg['lstm_ae']

        class LSTMAutoencoder(nn.Module):   # PHẢI giống hệt lớp dùng khi huấn luyện
            def __init__(self, seq_len, no_features, embedding_dim, num_layers, dropout_rate):
                super().__init__()
                self.seq_len = seq_len
                self.encoder_lstm = nn.LSTM(no_features, embedding_dim, num_layers=num_layers, batch_first=True,
                                            dropout=dropout_rate if num_layers > 1 else 0.0)
                self.decoder_lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers=num_layers, batch_first=True,
                                            dropout=dropout_rate if num_layers > 1 else 0.0)
                self.output_linear = nn.Linear(embedding_dim, no_features)

            def forward(self, x):
                _, (h, _) = self.encoder_lstm(x)
                latent = h[-1].unsqueeze(1).repeat(1, self.seq_len, 1)
                out, _ = self.decoder_lstm(latent)
                return self.output_linear(out)

        m = LSTMAutoencoder(self.W, len(self.features), la['embedding_dim'], la['num_layers'], la['dropout'])
        m.load_state_dict(torch.load(os.path.join(self.dir, la['file']), map_location='cpu'))
        m.eval()

        def recon(window):
            with torch.no_grad():
                x = torch.tensor(np.asarray(window, dtype=np.float32)[None, ...])
                return m(x)[0].numpy()
        return recon

    # ------------------------------------------------------------------ helpers
    def _pop_baseline(self, age):
        k = '60-69' if age < 70 else '70-79' if age < 80 else '80+'
        return self.pop[k]

    @staticmethod
    def _num(v):
        try:
            v = float(v)
            return v if v == v else None
        except (TypeError, ValueError):
            return None

    def _vi(self, f):
        return self.names_vi.get(f, f)

    @staticmethod
    def _std(s):
        sd = s.std()
        return sd if (sd == sd and sd > 1e-8) else 1.0

    # ------------------------------------------------------------------ main
    def predict(self, user_id, daily_history, age=None, custom_thresholds=None):
        import pandas as pd
        age = age if age is not None else self.cs['assumed_age_default']
        if not daily_history:
            return {'status': 'no_data', 'user_id': str(user_id)}

        g = pd.DataFrame(daily_history).copy()
        g['date'] = pd.to_datetime(g['date']).dt.normalize()
        for col in self.features + ['resting_hr_p10', 'SpO2_min']:
            g[col] = pd.to_numeric(g[col], errors='coerce') if col in g.columns else np.nan
        g['valid_day'] = g['valid_day'].fillna(False).astype(bool) if 'valid_day' in g.columns else False
        g = g.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)

        today = g.iloc[-1]
        if not today['heart_rate_mean'] == today['heart_rate_mean']:
            return {'status': 'no_heart_rate_today', 'user_id': str(user_id), 'date': str(today['date'].date())}

        prev = g.iloc[:-1]
        prev_valid = prev[prev['valid_day']]
        valid = bool(today['valid_day'])
        n_valid = len(prev_valid) + (1 if valid else 0)
        day = max(1, n_valid)
        cs = self.cs
        tier = 1 if day <= cs['tier1_days'][1] else 2 if day <= cs['tier2_days'][1] else 3

        level, why = 'NONE', []
        out = {'status': 'ok', 'user_id': str(user_id), 'date': str(today['date'].date()), 'tier': tier,
               'n_valid_days': int(n_valid), 'valid_day': valid, 'model_version': self.cfg['version'],
               'anomaly_score': None, 'ml': None, 'top_features': []}

        def up(l, msg):
            nonlocal level
            if LEVEL[l] > LEVEL[level]:
                level = l
            why.append(msg)

        # ---- Luật cứng (mọi tier, kể cả ngày thiếu dữ liệu)
        # ponytail: Nếu có ngưỡng riêng do bác sĩ chỉ định cho hồ sơ, thay thế ngưỡng lâm sàng mặc định
        cl = dict(self.clin)
        if custom_thresholds:
            if 'hr_high' in custom_thresholds and custom_thresholds['hr_high'] is not None:
                cl['hr_high'] = float(custom_thresholds['hr_high'])
            if 'hr_low' in custom_thresholds and custom_thresholds['hr_low'] is not None:
                cl['hr_low'] = float(custom_thresholds['hr_low'])
            if 'spo2_crit' in custom_thresholds and custom_thresholds['spo2_crit'] is not None:
                cl['spo2_crit'] = float(custom_thresholds['spo2_crit'])

        if today['heart_rate_mean'] > cl['hr_high']:
            up('CRITICAL', f"HR trung bình {today['heart_rate_mean']:.0f} > {cl['hr_high']}")
        if today['resting_hr_p10'] < cl['hr_low']:
            up('CRITICAL', f"HR nghỉ (p10) {today['resting_hr_p10']:.0f} < {cl['hr_low']}")
        sp = today['SpO2_min']
        if sp == sp:
            if sp < cl['spo2_crit']:
                up('CRITICAL', f"SpO2 min {sp:.0f} < {cl['spo2_crit']} (tham khảo)")
            elif sp < cl['spo2_warn'] and self.cfg.get('spo2_soft_enabled', False):
                up('SOFT', f"SpO2 min {sp:.0f} < {cl['spo2_warn']} (tham khảo)")
        out['rule_hits'] = list(why)

        z_soft = cs['soft_z']
        if tier == 1:
            # ponytail: Tầng 1 chỉ cần có nhịp tim là đánh giá được (ưu tiên ngưỡng bác sĩ gán riêng, nếu không dùng chuẩn tuổi)
            hr_val = today['resting_hr_p10'] if today['resting_hr_p10'] == today['resting_hr_p10'] else today['heart_rate_mean']
            if custom_thresholds and ('hr_high' in custom_thresholds or 'hr_low' in custom_thresholds):
                if hr_val > cl['hr_high']:
                    up('CRITICAL', f"Nhịp tim {hr_val:.0f} vượt ngưỡng tối đa của bác sĩ ({cl['hr_high']:.0f})")
                elif hr_val < cl['hr_low']:
                    up('CRITICAL', f"Nhịp tim {hr_val:.0f} dưới ngưỡng tối thiểu của bác sĩ ({cl['hr_low']:.0f})")
                out['anomaly_score'] = float(hr_val) if hr_val == hr_val else None
            else:
                pop = self._pop_baseline(age)
                z = (hr_val - pop['mean']) / pop['std'] if hr_val == hr_val else 0.0
                out['anomaly_score'] = float(z)
                if abs(z) > z_soft:
                    up('SOFT', f'HR lệch {z:+.1f}σ so với chuẩn tuổi (giả định {age})')
        elif not valid:
            why.append('Ngày chưa đủ dữ liệu (HR/steps/sleep) -> chưa xét baseline')
        elif tier == 2:
            h = prev_valid['resting_hr_p10'].dropna()
            if len(h) >= cs['min_days_baseline']:
                z = (today['resting_hr_p10'] - h.mean()) / self._std(h)
                out['anomaly_score'] = float(z)
                if abs(z) > z_soft:
                    up('SOFT', f'HR nghỉ lệch {z:+.1f}σ so với baseline tạm ({len(h)} ngày)')
        else:
            self._tier3(g, today, out, up, why)

        out['alert'] = level
        out['reason'] = '; '.join(why) if why else 'Bình thường'
        out['action'] = ('Liên hệ bác sĩ' if level == 'CRITICAL' else
                         'Theo dõi thêm' if level == 'SOFT' else 'Tiếp tục theo dõi')
        if tier == 2 and level != 'CRITICAL':
            out['action'] += f" ({n_valid}/{cs['tier2_days'][1] + 1} ngày để có baseline cá nhân)"
        return out

    # ------------------------------------------------------------------ tier 3
    def _tier3(self, g, today, out, up, why):
        import pandas as pd
        F, W = self.features, self.W
        vd = g[g['valid_day']]
        # z-score cá nhân: mean/std trên MỌI ngày hợp lệ tới hôm nay (giống personal_z lúc huấn luyện)
        z = pd.DataFrame({'date': g['date']})
        for f in F:
            s = vd[f].dropna()
            z[f] = (g[f] - s.mean()) / self._std(s) if len(s) >= self.cs['min_days_baseline'] else np.nan

        win = g.iloc[-W:]
        zw = z.iloc[-W:]
        ok = (len(win) == W and win['valid_day'].all() and not zw[F].isna().any().any()
              and (np.diff(win['date'].values) == np.timedelta64(1, 'D')).all())
        if not ok:
            out['ml'] = {'status': 'no_consecutive_window',
                         'detail': f'Cần {W} ngày hợp lệ liên tiếp kết thúc hôm nay; chỉ áp dụng luật cứng'}
            why.append(f'Chưa đủ {W} ngày hợp lệ liên tiếp -> bỏ qua LSTM-AE/IF')
            return

        X = zw[F].values.astype(np.float32)                     # (7, 6)
        rec = np.asarray(self.recon_fn(X), dtype=np.float32)
        sq = (rec - X) ** 2
        mse = float(sq.mean())
        err_feat = sq.mean(axis=0)                              # (6,) sai số tái tạo theo feature
        iso_score = float(-self.iso.decision_function(X.reshape(1, -1))[0])

        lstm_anom = mse > self.lstm_th
        iso_anom = iso_score > self.iso_th
        hybrid = bool(lstm_anom or iso_anom)

        total = float(err_feat.sum()) or 1.0
        order = np.argsort(-err_feat)[:3]
        top = [{'feature': F[i], 'name_vi': self._vi(F[i]), 'recon_error': float(err_feat[i]),
                'share': float(err_feat[i] / total), 'z_today': float(X[-1, i])} for i in order]

        out['anomaly_score'] = mse
        out['top_features'] = top
        out['ml'] = {'status': 'ok', 'window_end': str(today['date'].date()),
                     'lstm_mse': mse, 'lstm_threshold': self.lstm_th, 'lstm_anomaly': bool(lstm_anom),
                     'isof_score': iso_score, 'isof_threshold': self.iso_th, 'isof_anomaly': bool(iso_anom),
                     'hybrid_anomaly': hybrid}
        if hybrid:
            who = ' + '.join(n for n, a in (('LSTM-AE', lstm_anom), ('Isolation Forest', iso_anom)) if a)
            detail = ', '.join(f"{t['name_vi']} ({t['share']*100:.0f}% sai số, {t['z_today']:+.1f}σ hôm nay)" for t in top)
            up('SOFT', f'{who} báo bất thường (cửa sổ {W} ngày). Đóng góp chính: {detail}')


_PREDICTOR = {}


def predict_anomaly_week7(user_id, daily_history, age=None, custom_thresholds=None, package_dir=PACKAGE_DIR):
    """Hàm BE gọi. Xem docstring đầu file và inference_spec_week7.md."""
    p = _PREDICTOR.get(package_dir)
    if p is None:
        p = _PREDICTOR[package_dir] = Predictor(package_dir)
    return p.predict(user_id, daily_history, age, custom_thresholds)
