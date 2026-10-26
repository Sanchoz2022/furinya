//! Политика инициативы: таймер НЕ диктует, а лишь даёт шанс.
//!
//! Правило Саши: «не постоянно, ровно как по часам, а как бы модель сама
//! выбирала, надо ответить или нет». Поэтому:
//! - пауза до шанса каждый раз случайная внутри окна (не «как по часам»);
//! - после СВОЕГО сообщения пауза длиннее (человек не пишет «а ты тут?»
//!   сразу после того, как уже написал);
//! - количество своих сообщений без ответа ограничено (страховка от спама),
//!   но НЕ навсегда: после серии — длинная пауза (~3 часа) и новый шанс;
//! - «заснула» = только тихие часы (он спит). Долгая тишина в активные часы
//!   НЕ усыпляет бота: иначе инициатива умирает до его следующего сообщения.

use rand::Rng;

/// Длинная пауза после серии своих сообщений без ответа. Не «навсегда»:
/// человек, которому не ответили, не замолкает пожизненно — он пробует снова
/// позже (например, на следующий день с другой мыслью).
pub const LONG_COOLDOWN_SECS: u64 = 3 * 60 * 60;

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

    /// Пауза после исчерпания лимита сообщений без ответа: ~3 часа + разброс,
    /// чтобы «проснуться» не выглядело ровно по часам.
    pub fn long_cooldown(&self) -> u64 {
        let mut rng = rand::thread_rng();
        rng.gen_range(LONG_COOLDOWN_SECS..=LONG_COOLDOWN_SECS + self.window_max)
    }
}

/// Тихие часы: окно, когда юзер обычно спит.
/// В это время бот НЕ проявляет инициативу, не пишет «спокойной ночи» и
/// не будит сообщениями — только тихо консолидирует память.
/// Окно может переходить через полночь (23 → 7).
#[derive(Debug, Clone, Copy)]
pub struct QuietHours {
    /// Начало (час по локальному времени). 0 = выключено.
    pub start: u32,
    /// Конец (час, НЕ включается). 0 = выключено.
    pub end: u32,
}

impl QuietHours {
    pub fn new(start: u32, end: u32) -> Self {
        let start = start.min(23);
        let end = end.min(23);
        Self { start, end }
    }

    /// Включены ли тихие часы (start != end).
    pub fn enabled(&self) -> bool {
        self.start != self.end
    }

    /// В тихие ли часы сейчас (час по локальному времени, 0..23).
    pub fn is_quiet_now(&self, hour: u32) -> bool {
        if !self.enabled() {
            return false;
        }
        let h = hour.min(23);
        if self.start < self.end {
            h >= self.start && h < self.end
        } else {
            // окно через полночь: 23..7 → h >= 23 || h < 7
            h >= self.start || h < self.end
        }
    }
}

/// Событие смены состояния «тихие часы» — вынесено, чтобы логика
/// «заснула / проснулась» была покрыта тестами.
#[derive(Debug, PartialEq, Eq)]
pub enum QuietEvent {
    /// Ничего не изменилось.
    None,
    /// Вошли в тихие часы: он спит — замолчали.
    Entered,
    /// Вышли из тихих часов: проснулись, инициатива снова разрешена.
    Woke,
}

/// Следит за окном тихих часов и сообщает о сменах состояния.
#[derive(Debug, Clone, Copy, Default)]
pub struct QuietWatcher {
    was_quiet: bool,
}

impl QuietWatcher {
    pub fn new() -> Self {
        Self { was_quiet: false }
    }

    /// Состояние окна на данный момент (для логов и решений).
    pub fn event(&mut self, quiet_now: bool) -> QuietEvent {
        if quiet_now && !self.was_quiet {
            self.was_quiet = true;
            QuietEvent::Entered
        } else if !quiet_now && self.was_quiet {
            self.was_quiet = false;
            QuietEvent::Woke
        } else {
            QuietEvent::None
        }
    }

    /// Находимся ли сейчас в тихих часах.
    pub fn in_quiet(&self) -> bool {
        self.was_quiet
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watcher_reports_enter_and_wake() {
        let mut w = QuietWatcher::new();
        assert_eq!(w.event(false), QuietEvent::None);
        assert_eq!(w.event(true), QuietEvent::Entered);
        assert_eq!(w.event(true), QuietEvent::None, "повторно не событие");
        assert!(w.in_quiet());
        assert_eq!(w.event(false), QuietEvent::Woke);
        assert_eq!(w.event(false), QuietEvent::None);
        assert!(!w.in_quiet());
    }

    #[test]
    fn watcher_started_inside_quiet_wakes_once() {
        let mut w = QuietWatcher::new();
        // Бот стартовал ночью: первый тик — тихие часы, потом рассвет.
        assert_eq!(w.event(true), QuietEvent::Entered);
        assert_eq!(w.event(true), QuietEvent::None);
        assert_eq!(w.event(false), QuietEvent::Woke);
    }

    #[test]
    fn quiet_hours_basic_window() {
        let q = QuietHours::new(2, 9);
        assert!(q.enabled());
        assert!(!q.is_quiet_now(1));
        assert!(q.is_quiet_now(2));
        assert!(q.is_quiet_now(5));
        assert!(q.is_quiet_now(8));
        assert!(!q.is_quiet_now(9));
        assert!(!q.is_quiet_now(14));
    }

    #[test]
    fn quiet_hours_across_midnight() {
        let q = QuietHours::new(23, 7);
        assert!(q.is_quiet_now(23));
        assert!(q.is_quiet_now(2));
        assert!(q.is_quiet_now(6));
        assert!(!q.is_quiet_now(7));
        assert!(!q.is_quiet_now(12));
    }

    #[test]
    fn quiet_hours_disabled() {
        let q = QuietHours::new(0, 0);
        assert!(!q.enabled());
        for h in 0..24 {
            assert!(!q.is_quiet_now(h));
        }
        // start == end — тоже выключено
        let q2 = QuietHours::new(5, 5);
        assert!(!q2.enabled());
        assert!(!q2.is_quiet_now(5));
    }

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
    fn long_cooldown_is_three_hours_not_forever() {
        let p = IdlePolicy::new(300, 900, 3);
        for _ in 0..50 {
            let d = p.long_cooldown();
            assert!(
                (LONG_COOLDOWN_SECS..=LONG_COOLDOWN_SECS + 900).contains(&d),
                "кулдаун {d} вне окна"
            );
            assert!(d > 0, "кулдаун не должен быть нулевым/вечным");
        }
    }

    #[test]
    fn pings_are_reset_after_long_cooldown() {
        // После серии без ответа политика снова разрешает шанс — не навсегда.
        let p = IdlePolicy::new(300, 900, 3);
        assert!(!p.may_offer(false, 3));
        assert!(p.may_offer(false, 0), "после сброса счётчика шанс снова есть");
    }

    #[test]
    fn windows_are_sanitized() {
        let p = IdlePolicy::new(0, 10, 0);
        assert!(p.window_min >= 5);
        assert!(p.window_max >= p.window_min);
        assert!(p.ping_max >= 1);
    }
}
