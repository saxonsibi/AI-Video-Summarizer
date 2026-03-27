import { motion } from 'framer-motion'
import { Brain, CheckCircle2, Film, Star } from 'lucide-react'

function normalizeWhitespace(value) {
  return String(value || '')
    .replace(/\s+/g, ' ')
    .trim()
}

function normalizeTimestamp(value) {
  const raw = normalizeWhitespace(value)
  if (!raw) return '00:00'

  const matched = raw.match(/\d{1,2}:\d{2}(?::\d{2})?/)
  if (!matched) return '00:00'
  return matched[0]
}

function cleanSentenceSpacing(value) {
  return normalizeWhitespace(value)
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([.!?])([A-Za-z])/g, '$1 $2')
    .replace(/([a-z])(\d{1,2}:\d{2}(?::\d{2})?)/g, '$1 $2')
    .replace(/\s*[-|]+\s*/g, ' ')
    .trim()
}

function cleanChapterTitle(chapter) {
  const title = cleanSentenceSpacing(chapter?.title)
    .replace(/^\d{1,2}:\d{2}(?::\d{2})?\s*/, '')
    .replace(/^[—:-]\s*/, '')
    .trim()

  if (!title) return 'Untitled chapter'
  if (title.length <= 72) return title
  return `${title.slice(0, 69).trim()}...`
}

function cleanListItem(value) {
  return cleanSentenceSpacing(value)
    .replace(/^[•*-]\s*/, '')
    .trim()
}

function SummarySectionCard({ icon: Icon, title, accentClass, children }) {
  return (
    <motion.section
      className="summary-dashboard-card"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <div className="flex items-start gap-4 mb-5">
        <div className={`summary-dashboard-icon ${accentClass}`}>
          <Icon className="w-5 h-5" />
        </div>
        <div className="min-w-0">
          <h3 className="summary-dashboard-title">{title}</h3>
        </div>
      </div>
      <div className="summary-dashboard-content">{children}</div>
    </motion.section>
  )
}

function SummaryDashboard({ summary, onJumpToTimestamp }) {
  const safeSummary = summary || {
    tldr: '',
    key_points: [],
    action_items: [],
    chapters: [],
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      <SummarySectionCard icon={Brain} title="TLDR" accentClass="summary-dashboard-icon-brain">
        {safeSummary.tldr ? (
          <p className="summary-dashboard-text summary-dashboard-tldr">
            {cleanSentenceSpacing(safeSummary.tldr)}
          </p>
        ) : (
          <p className="summary-dashboard-empty">No TLDR available yet.</p>
        )}
      </SummarySectionCard>

      <SummarySectionCard icon={Star} title="Key Points" accentClass="summary-dashboard-icon-star">
        {safeSummary.key_points?.length ? (
          <ul className="space-y-3">
            {safeSummary.key_points.map((point, index) => (
              <li key={`key-point-${index}`} className="summary-dashboard-bullet">
                <span className="summary-dashboard-bullet-dot" />
                <span className="summary-dashboard-text">{cleanListItem(point)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="summary-dashboard-empty">No key points available yet.</p>
        )}
      </SummarySectionCard>

      <SummarySectionCard icon={CheckCircle2} title="Action Items" accentClass="summary-dashboard-icon-check">
        {safeSummary.action_items?.length ? (
          <ul className="space-y-3">
            {safeSummary.action_items.map((item, index) => (
              <li key={`action-item-${index}`} className="summary-dashboard-check-row">
                <span className="summary-dashboard-checkbox">
                  <CheckCircle2 className="w-4 h-4" />
                </span>
                <span className="summary-dashboard-text">{cleanListItem(item)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="summary-dashboard-empty">No direct action items detected.</p>
        )}
      </SummarySectionCard>

      <SummarySectionCard icon={Film} title="Chapters" accentClass="summary-dashboard-icon-film">
        {safeSummary.chapters?.length ? (
          <div className="space-y-3">
            {safeSummary.chapters.map((chapter, index) => (
              <button
                key={`chapter-${index}`}
                type="button"
                className="summary-dashboard-chapter"
                onClick={() => onJumpToTimestamp?.(normalizeTimestamp(chapter?.timestamp))}
              >
                <span className="summary-dashboard-chapter-line">
                  <span className="summary-dashboard-timestamp">{normalizeTimestamp(chapter?.timestamp)}</span>
                  <span className="summary-dashboard-chapter-separator">—</span>
                  <span className="summary-dashboard-chapter-title">{cleanChapterTitle(chapter)}</span>
                </span>
              </button>
            ))}
          </div>
        ) : (
          <p className="summary-dashboard-empty">No chapters available yet.</p>
        )}
      </SummarySectionCard>
    </div>
  )
}

export default SummaryDashboard
