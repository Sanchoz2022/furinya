//! Политика инициативы: таймер НЕ диктует, а лишь даёт шанс.
//!
//! Правило Саши: «не постоянно, ровно как по часам, а как бы модель сама
//! выбирала, надо ответить или нет». Поэтому:
//! - пауза до шанса каждый раз случайная внутри окна (не «как по часам»);
//! - после СВОЕГО сообщения пауза длиннее (человек не пишет «а ты тут?»
//!   сразу после того, как уже написал);
//! - количество своих сообщений без ответа ограничено (страховка от спама);
//! - «заснула» — первыми не пишем вообще, пока не напишет он.

use rand::Rng;

/// Политика инициативы (чистые правила, без IO — чинится юнит-тестами).
#[derive(Debug, Clone, Copy)]
pub struct IdlePolicy {
    /// Минимальная пауза до шанса написать первой (сек).
    pub window_min: u64,
    /// Максимальная пауза до шанса (сек).
    pub window_max: u64,
    /// Максимум своих сообщений подряд без ответа.
    pub ping_max: u32,
}

impl IdlePolicy {
    pub fn new(window_min: u64, window_max: u64, ping_max: u32) -> Self {
        // Пол 5с (не 60): чтобы ускоренные прогоны с коротким окном работали.
        let window_min = window_min.max(5);
        let window_max = window_max.max(window_min);
        Self {
            window_min,
            window_max,
            ping_max: ping_max.max(1),
        }
    }

    /// Пауза до следующего шанса предложить написать первой.
    /// `after_own_message` = мы только что сами написали → пауза длиннее.
    pub fn next_delay(&self, after_own_message: bool) -> u64 {
        let mut rng = rand::thread_rng();
        if after_own_message {
            // 2× окно: своё сообщение «съедает» интерес к диалогу.
            rng.gen_range(2 * self.window_min..=2 * self.window_max)
        } else {
            rng.gen_range(self.window_min..=self.window_max)
        }
    }

    /// Можно ли сейчас предлагать шанс написать первой.
    pub fn may_offer(&self, asleep: bool, pings: u32) -> bool {
        !asleep && pings < self.ping_max
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_within_window() {
        let p = IdlePolicy::new(300, 900, 3);
        for _ in 0..200 {
            let d = p.next_delay(false);
            assert!((300..=900).contains(&d), "задержка {d} вне окна");
        }
    }

    #[test]
    fn delay_after_own_message_is_longer() {
        let p = IdlePolicy::new(300, 900, 3);
        for _ in 0..200 {
            let d = p.next_delay(true);
            assert!((600..=1800).contains(&d), "задержка {d} вне окна 2x");
        }
    }

    #[test]
    fn delay_is_random_not_constant() {
        // Не «ровно как по часам»: задержки должны различаться.
        let p = IdlePolicy::new(300, 900, 3);
        let mut uniq = std::collections::HashSet::new();
        for _ in 0..50 {
            uniq.insert(p.next_delay(false));
        }
        assert!(uniq.len() > 10, "слишком мало разных задержек: {}", uniq.len());
    }

    #[test]
    fn may_offer_respects_sleep_and_ping_limit() {
        let p = IdlePolicy::new(300, 900, 3);
        assert!(p.may_offer(false, 0));
        assert!(p.may_offer(false, 2));
        assert!(!p.may_offer(false, 3), "после лимита пингов — тишина");
        assert!(!p.may_offer(true, 0), "во сне первыми не пишем");
    }

    #[test]
    fn windows_are_sanitized() {
        let p = IdlePolicy::new(0, 10, 0);
        assert!(p.window_min >= 5);
        assert!(p.window_max >= p.window_min);
        assert!(p.ping_max >= 1);
    }
}
